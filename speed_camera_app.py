import sys
import time
import io
import cv2
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTableView, 
                             QHeaderView, QFileDialog, QMessageBox, QFrame,
                             QComboBox) # --- NEW ---
from PyQt5.QtGui import QImage, QPixmap, QStandardItemModel, QStandardItem, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot

# --- PDF Generation ---
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# =============================================================================
#  STYLESHEET - Darkish Blue Theme
# =============================================================================
dark_stylesheet = """
    QWidget {
        background-color: #2C3E50;
        color: #ECF0F1;
        font-family: Segoe UI;
        font-size: 10pt;
    }
    QMainWindow {
        border-image: none;
    }
    QLabel {
        color: #ECF0F1;
    }
    QPushButton, QComboBox {
        background-color: #3498DB;
        color: #ECF0F1;
        border: 1px solid #2980B9;
        padding: 8px;
        border-radius: 4px;
    }
    QPushButton:hover, QComboBox:hover {
        background-color: #4Ea2e0;
    }
    QPushButton:pressed {
        background-color: #2980B9;
    }
    QComboBox::drop-down {
        border: none;
    }
    QComboBox QAbstractItemView {
        background-color: #34495E;
        color: #ECF0F1;
        selection-background-color: #3498DB;
    }
    QTableView {
        background-color: #34495E;
        border: 1px solid #2C3E50;
        gridline-color: #2C3E50;
    }
    QHeaderView::section {
        background-color: #3498DB;
        color: white;
        padding: 4px;
        border: 1px solid #2C3E50;
    }
    QFrame#video_frame {
        border: 2px solid #3498DB;
        border-radius: 5px;
    }
"""

# =============================================================================
#  CAMERA AND PROCESSING THREAD
# =============================================================================
class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    speed_detected = pyqtSignal(float, QPixmap)

    # --- MODIFIED ---: Accept orientation in the constructor
    def __init__(self, parent=None, orientation="Horizontal"):
        super().__init__(parent)
        self.running = False
        self.camera = None
        self.orientation = orientation
        
        # --- CALIBRATION ---
        self.DISTANCE_METERS = 0.3

        # --- DETECTION PARAMETERS ---
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=50, detectShadows=False)
        self.min_contour_area = 500
        
        # --- State Machine for Detection ---
        self.object_state = "WAITING_FOR_LINE_1"
        self.start_time = 0
        self.snapshot = None

    def run(self):
        self.running = True
        self.camera = cv2.VideoCapture(0)
        
        if not self.camera.isOpened():
            print("Error: Cannot open camera.")
            self.running = False
            return

        while self.running:
            ret, frame = self.camera.read()
            if ret:
                height, width, _ = frame.shape
                
                # --- MODIFIED ---: Handle both orientations for processing and drawing
                processed_frame = frame.copy()

                if self.orientation == "Horizontal":
                    line1_pos = int(height * 0.4)
                    line2_pos = int(height * 0.6)
                    processed_frame = self.process_frame(processed_frame, line1_pos, line2_pos)
                    # Draw lines
                    cv2.line(processed_frame, (0, line1_pos), (width, line1_pos), (0, 255, 0), 2)
                    cv2.line(processed_frame, (0, line2_pos), (width, line2_pos), (0, 0, 255), 2)
                
                elif self.orientation == "Vertical":
                    line1_pos = int(width * 0.4)
                    line2_pos = int(width * 0.6)
                    processed_frame = self.process_frame(processed_frame, line1_pos, line2_pos)
                    # Draw lines
                    cv2.line(processed_frame, (line1_pos, 0), (line1_pos, height), (0, 255, 0), 2)
                    cv2.line(processed_frame, (line2_pos, 0), (line2_pos, height), (0, 0, 255), 2)

                self.frame_ready.emit(processed_frame)
            
            time.sleep(0.03)

        self.camera.release()

    # --- MODIFIED ---: Process frame based on orientation
    def process_frame(self, frame, line1_pos, line2_pos):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        fg_mask = self.bg_subtractor.apply(blurred)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.drawContours(frame, [contour], -1, (255, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 7, (255, 0, 255), -1)

            # --- MODIFIED ---: Use a single detection logic that adapts
            # Determine which coordinate to check based on orientation
            check_coord = cy if self.orientation == "Horizontal" else cx
            
            # State Machine Logic
            if self.object_state == "WAITING_FOR_LINE_1" and (line1_pos - 10 < check_coord < line1_pos + 10):
                self.start_time = time.time()
                self.snapshot = self.convert_cv_to_qpixmap(frame)
                self.object_state = "WAITING_FOR_LINE_2"
                print(f"Object crossed line 1 (at {check_coord})")

            elif self.object_state == "WAITING_FOR_LINE_2" and (line2_pos - 10 < check_coord < line2_pos + 10):
                end_time = time.time()
                elapsed_time = end_time - self.start_time
                
                if elapsed_time > 0.05:
                    speed_mps = self.DISTANCE_METERS / elapsed_time
                    speed_kmh = speed_mps * 3.6
                    print(f"Speed: {speed_kmh:.2f} km/h")
                    self.speed_detected.emit(speed_kmh, self.snapshot)
                
                self.object_state = "WAITING_FOR_LINE_1"
                self.snapshot = None

        return frame

    def convert_cv_to_qpixmap(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    def stop(self):
        self.running = False
        self.wait()

# =============================================================================
#  MAIN APPLICATION WINDOW
# =============================================================================
class SpeedCamApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speed Detection Camera")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        left_panel = QVBoxLayout()
        self.video_label = QLabel("Camera is OFF")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setObjectName("video_frame")
        self.video_label.setMinimumSize(640, 480)
        left_panel.addWidget(self.video_label)
        
        right_panel = QVBoxLayout()
        
        # --- NEW ---: Configuration Layout with Dropdown
        config_layout = QHBoxLayout()
        config_label = QLabel("Line Orientation:")
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Horizontal", "Vertical"])
        config_layout.addWidget(config_label)
        config_layout.addWidget(self.orientation_combo)
        right_panel.addLayout(config_layout)
        right_panel.addSpacing(10)

        # Control Buttons
        controls_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Camera")
        self.stop_button = QPushButton("Stop Camera")
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        right_panel.addLayout(controls_layout)
        right_panel.addSpacing(20)
        
        # Results Table
        self.results_table = QTableView()
        self.results_model = QStandardItemModel()
        self.results_model.setHorizontalHeaderLabels(["Timestamp", "Speed (km/h)", "Snapshot"])
        self.results_table.setModel(self.results_model)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setColumnWidth(2, 150)
        self.results_table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.results_table.verticalHeader().setDefaultSectionSize(100)
        self.results_table.setEditTriggers(QTableView.NoEditTriggers)
        right_panel.addWidget(self.results_table)
        
        # Export/Clear buttons for results table
        export_clear_layout = QHBoxLayout()
        self.clear_button = QPushButton("Clear Results")
        self.export_button = QPushButton("Export to PDF")
        export_clear_layout.addWidget(self.clear_button)
        export_clear_layout.addWidget(self.export_button)
        right_panel.addLayout(export_clear_layout)
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        self.start_button.clicked.connect(self.start_camera)
        self.stop_button.clicked.connect(self.stop_camera)
        self.export_button.clicked.connect(self.export_to_pdf)
        self.clear_button.clicked.connect(self.clear_results)
        
        self.camera_thread = None

    @pyqtSlot()
    def start_camera(self):
        if not self.camera_thread or not self.camera_thread.isRunning():
            # --- MODIFIED ---: Get orientation from combo box and pass to thread
            selected_orientation = self.orientation_combo.currentText()
            self.camera_thread = CameraThread(self, orientation=selected_orientation)
            
            self.camera_thread.frame_ready.connect(self.update_video_frame)
            self.camera_thread.speed_detected.connect(self.add_speed_entry)
            self.camera_thread.start()
            
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.orientation_combo.setEnabled(False) # --- NEW ---: Disable combo box
            self.video_label.setText("Starting camera...")

    @pyqtSlot()
    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.orientation_combo.setEnabled(True) # --- NEW ---: Re-enable combo box
            self.video_label.setText("Camera is OFF")
            self.video_label.setPixmap(QPixmap())

    @pyqtSlot(np.ndarray)
    def update_video_frame(self, cv_img):
        qt_img = self.convert_cv_to_qpixmap(cv_img)
        self.video_label.setPixmap(qt_img.scaled(self.video_label.size(), 
                                                Qt.KeepAspectRatio, 
                                                Qt.SmoothTransformation))

    def convert_cv_to_qpixmap(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(convert_to_Qt_format)

    @pyqtSlot(float, QPixmap)
    def add_speed_entry(self, speed, snapshot_pixmap):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        speed_str = f"{speed:.2f}"
        
        timestamp_item = QStandardItem(timestamp)
        speed_item = QStandardItem(speed_str)
        
        thumbnail_item = QStandardItem()
        thumbnail = snapshot_pixmap.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        thumbnail_item.setIcon(QIcon(thumbnail))
        
        self.results_model.appendRow([timestamp_item, speed_item, thumbnail_item])
        self.results_table.scrollToBottom()

    @pyqtSlot()
    def clear_results(self):
        self.results_model.clear()
        self.results_model.setHorizontalHeaderLabels(["Timestamp", "Speed (km/h)", "Snapshot"])

    @pyqtSlot()
    def export_to_pdf(self):
        if self.results_model.rowCount() == 0:
            QMessageBox.warning(self, "Export Error", "There are no results to export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", "", "PDF Files (*.pdf)")
        
        if path:
            try:
                doc = SimpleDocTemplate(path, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []

                title = Paragraph("Speed Detection Report", styles['h1'])
                story.append(title)
                story.append(Spacer(1, 0.2 * inch))

                table_data = [["Timestamp", "Speed (km/h)", "Snapshot"]]
                for row in range(self.results_model.rowCount()):
                    timestamp = self.results_model.item(row, 0).text()
                    speed = self.results_model.item(row, 1).text()
                    
                    icon = self.results_model.item(row, 2).icon()
                    pixmap = icon.pixmap(icon.actualSize(QSize(120,90)))
                    
                    buffer = io.BytesIO()
                    pixmap.save(buffer, "PNG")
                    buffer.seek(0)
                    
                    img = Image(buffer, width=1.2*inch, height=0.9*inch)
                    table_data.append([timestamp, speed, img])

                report_table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
                style = TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498DB")),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#34495E")),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#2C3E50"))
                ])
                report_table.setStyle(style)
                
                story.append(report_table)
                doc.build(story)
                
                QMessageBox.information(self, "Success", f"Report successfully saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting to PDF:\n{e}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

# =============================================================================
#  APPLICATION ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    from PyQt5.QtCore import QSize
    app = QApplication(sys.argv)
    app.setStyleSheet(dark_stylesheet)
    window = SpeedCamApp()
    window.show()
    sys.exit(app.exec_())