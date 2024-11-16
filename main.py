from PyQt5.QtWidgets import QApplication,QMainWindow
from PyQt5 import uic
from utils.category_functions import add_category, delete_category
from utils.file_functions_sam import ImageViewer
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QSizePolicy

Ui_MainWindow, BaseClass = uic.loadUiType("main.ui")


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Configure the layout
        self.initUI()

        # Set fixed widths - make buttons narrower
        total_width = 200  # Total width for category list and buttons
        button_width = total_width // 2  # Each button gets half

        # Configure category list to match total width of buttons
        self.listWidget.setFixedWidth(total_width)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.listWidget.clear()  # Clear default categories

        # Add default numeric categories
        self.listWidget.addItem("1")
        self.listWidget.addItem("2")

        # Select first category by default
        self.listWidget.setCurrentRow(0)

        # Configure buttons to be narrower and align with list
        self.addButton.setFixedWidth(button_width)
        self.deleteButton.setFixedWidth(button_width)

        # Configure image list
        self.listWidget3.setMinimumWidth(250)
        self.listWidget3.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        # Create the ImageViewer instance
        self.myGraphicsViewInstance = ImageViewer(self.listWidget3, self.listWidget,
                                                  self.listWidget2, self.textEdit,
                                                  self.graphicsView)

        # Connect the Delete Annotation button
        self.deleteButton2.clicked.connect(self.myGraphicsViewInstance.deleteAllPoints)

    def initUI(self):
        # Connect buttons
        self.addButton.clicked.connect(lambda: add_category(self.listWidget))
        self.deleteButton.clicked.connect(lambda: delete_category(self.listWidget))
        self.openFile.triggered.connect(lambda: self.myGraphicsViewInstance.openImageFolder())
        self.saveFile.triggered.connect(lambda: self.myGraphicsViewInstance.saveAnnotationsToFile())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_E:
            self.myGraphicsViewInstance.isPointButtonClicked = False
            self.myGraphicsViewInstance.graphicsView.setCursor(QCursor(Qt.ArrowCursor))
            self.myGraphicsViewInstance.finishAnnotation()
        elif event.key() == Qt.Key_Z:
            self.myGraphicsViewInstance.undoLastPoint()
        elif event.key() == Qt.Key_Space:
            self.moveToNextImage()
            event.accept()

    def moveToNextImage(self):
        # Use listWidget3 (image list) instead of listWidget (category list)
        current_row = self.listWidget3.currentRow()
        if current_row < self.listWidget3.count() - 1:  # If not the last item
            next_row = current_row + 1
            self.listWidget3.setCurrentRow(next_row)
            # Trigger the display of the new image
            next_item = self.listWidget3.item(next_row)
            if next_item:
                self.myGraphicsViewInstance.displayImage(next_item, init_point=True)

if __name__ == '__main__':
    app = QApplication([])
    win = MainWindow()
    
    win.show()
    app.exec_()

