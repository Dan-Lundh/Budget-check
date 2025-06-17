import sys
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QWidget, QComboBox, QTableView, QHBoxLayout, QPushButton, QLabel,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtGui import QImage, QPixmap, QColor, QPainter, QBrush, QPen
from PySide6.QtCore import Qt, QBuffer, QIODevice, Signal
from keras._tf_keras.keras.models import load_model
import qimage2ndarray
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.constants import ROOTLIB
from utils.report_util import calc_per_date, calc_ratio, norm_image, join_data, arrayimage, filter
from utils.report_util import numpy_to_qimage, rgb_to_grayscale_weighted



# agent=Agent()

class ColorCircleWidget(QWidget):
    """
    A custom widget that displays a colored circle.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(50, 50) 
        self._state = 0 

    def set_state(self, new_state):
        if self._state != new_state:
            self._state = new_state
            self.update() # Trigger a repaint 

    def paintEvent(self, event):
        """
        Called when the widget needs to be repainted.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) 

        painter.setPen(QPen(Qt.black, 2)) # border, 2 pixels
        if self._state == 1:
            painter.setBrush(QBrush(Qt.red)) # Red  1
        else: # self._state == 0
            painter.setBrush(QBrush(Qt.green)) # Green  0

        # Calculate the size and position of the circle
        # We want it centered and fit within the widget
        circle_radius = min(self.width(), self.height()) / 2 - 5 
        center_x = self.width() / 2
        center_y = self.height() / 2

        # Draw the ellipse
        painter.drawEllipse(int(center_x - circle_radius),
                            int(center_y - circle_radius),
                            int(circle_radius * 2),
                            int(circle_radius * 2))

        painter.end() 

class BudgetGUI(QWidget):
    color_signal = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Budget Viewer")
        self.setGeometry(100, 100, 800, 600)
        self.key_column = 'Project'
        self.display_columns = ["Report Date", "Category", "Man-Hours (Est)", "Man-Hours (Actual)"]

        # Layout setup
        layout = QHBoxLayout(self) # main layout
        layout_left = QVBoxLayout()
        layout_right = QVBoxLayout()
        layout_right.setSpacing(0)
        image_layout = QHBoxLayout()
        excel_file = ROOTLIB / 'database' / 'Budget-demo.xlsx'
        self.df = pd.read_excel(excel_file)
        self.df.columns = self.df.columns.str.strip()  

        # Combo box setup
        self.combo = QComboBox()
        self.combo.addItems(sorted(self.df[self.key_column].dropna().unique()))
        self.combo.currentTextChanged.connect(self.show_details)

        # Table view setup
        self.table_view = QTableView()
        self.model = QStandardItemModel(self)
        self.table_view.setModel(self.model)

        # check setup
        self.button = QPushButton("Check for deviations")
        self.button.clicked.connect(self.budget_check)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter) 
        #self.image_label.setScaledContents(True)
        image_layout.addWidget(self.image_label)
        layout_right.addWidget(self.image_label)

        # Add widgets to layout
        layout_left.addWidget(self.combo)
        layout_left.addWidget(self.table_view)
        container_left = QWidget()
        container_left.setLayout(layout_left)
        container_right = QWidget()
        self.text = QLabel(f" Task by budget                  | Task to all tasks \n DE   Ma   QA   Ux   Tech     | DE   Ma   QA   Ux   Tech")
        layout_right.addWidget(self.text)
        layout_right.addWidget(self.button)
        container_right.setLayout(layout_right)

        layout.addWidget(container_left, stretch=1)
        layout.addWidget(container_right, stretch=1)

        # signal circle
        self.circle_widget = ColorCircleWidget()
        layout_right.addWidget(self.circle_widget, alignment=Qt.AlignCenter)
        self.color_signal.connect(self.circle_widget.set_state)
        if self.combo.count():
            self.show_details(self.combo.currentText())

    def show_details(self, selected_value):
        filtered = self.df[self.df[self.key_column] == selected_value]
        df_filtered = filtered[filtered["Category"] != "Total Project Est:"].copy()
        filtered = df_filtered[self.display_columns].dropna()

        pivot_df = filtered.pivot_table(index='Category', columns='Report Date', values=["Man-Hours (Est)", "Man-Hours (Actual)"])
        
        pivot_df.columns = [f"{col[0]}_{col[1]}" for col in pivot_df.columns]
        
        self.model.clear()
        cols=['Category']
        cols.extend(list(pivot_df.columns))
        self.model.setHorizontalHeaderLabels(cols)

        for row in pivot_df.itertuples(index=True):
            items = [QStandardItem(str(cell)) for cell in row]
            self.model.appendRow(items)

    def fix_input(self, selected_value):
        datumen=self.df[self.df["Project"]==selected_value]['Report Date'].dropna().unique()
        #print(datumen)
        data_df, datum2 = filter(self.df,selected_value)
        per_date=calc_per_date(data_df)
        per_ratio=calc_ratio(data_df)
        joined_df=join_data(norm_image(per_date), norm_image(per_ratio))
        i = len(datumen)  
        output=[]
        for t in range(i-2):
            img=[]
            for z in range(0+t,3+t,1):
                img.append(joined_df[z]*255)
            if len(img)>0:
                output.append(arrayimage(img))
        return output

    def display_image(self, np_array):
        q_image = numpy_to_qimage(np_array)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio,      
        )

        self.image_label.setPixmap(scaled_pixmap)

    def budget_check(self):
        selected_value=self.combo.currentText()
        model = self.load_fall_model() 
        img=np.array(self.fix_input(selected_value))
        y_pred = model.predict(img) 
        y_pred = np.argmax(y_pred, axis = 1)
        print(y_pred)
        array_original=np.array(["Project Management",
                                 "RL Development",
                                 "Data Engineering",
                                 "UI/UX (Mock)",
                                 "Testing & QA"])
        fpath= ROOTLIB / "database/normal_projects.npy"
        normals_array = np.load(fpath)
        normals_array_flat = normals_array.reshape(normals_array.shape[0], -1)
        for image in img:

            img_flat = image.reshape(-1)
            distances = np.linalg.norm(normals_array_flat - img_flat, axis=1)
            closest_index = np.argmin(distances)
            
            diff_image = rgb_to_grayscale_weighted(normals_array[closest_index]) - rgb_to_grayscale_weighted(image)
            diff_image2 = normals_array[closest_index] - image
            change_image= np.clip(diff_image, 0, 255)

            # change_image contains diff actual and normal (B is normal W is deviation)
            self.display_image(np.squeeze(np.clip(diff_image2, 0, 255)))
        self.color_signal.emit(y_pred[-1])
        
        
    def load_fall_model(self):
        fpath= ROOTLIB/"database/model.keras"
        return load_model(fpath)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BudgetGUI()
    window.show()
    sys.exit(app.exec())