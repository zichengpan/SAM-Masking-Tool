from PyQt5.QtWidgets import QInputDialog

def add_category(listWidget):
    text, ok = QInputDialog.getText(None, 'Add Category', 'Enter category name:')
    if ok and text:
        listWidget.addItem(text)

def delete_category(listWidget):
    current_item = listWidget.currentItem()
    if current_item:
        row = listWidget.row(current_item)
        listWidget.takeItem(row)
        print(listWidget)

def updateListWidget2(self):
    self.listWidget2.clear()

    for annotation in self.annotationStack:
        label = annotation['label']
        item = QListWidgetItem(label)
        self.listWidget2.addItem(item)

