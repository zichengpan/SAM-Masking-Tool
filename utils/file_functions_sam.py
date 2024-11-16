import numpy as np
from PyQt5.QtWidgets import (QFileDialog, QListWidgetItem, QGraphicsScene,
                             QGraphicsPixmapItem, QGraphicsView, QGraphicsPolygonItem,QGraphicsEllipseItem)
from PyQt5.QtGui import QColor, QPixmap,QPen, QBrush, QPolygonF,QCursor
from PyQt5.QtCore import QPointF,Qt
from PIL.ImageQt import ImageQt
from PIL import Image
import json
import cv2
import os
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# OpenCV installation for python >3.7
# pip install opencv-python-headless


class ImageViewer():
    def __init__(self, listWidget, categoryList, annList, textEdit, graphicsView, parent=None):
        self.listWidget = listWidget
        self.categoryList = categoryList
        self.annList = annList
        self.textEdit = textEdit
        self.graphicsView = graphicsView
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.pixmapItem = None
        self.folderPath = ""
        self.maskFolderPath = ""
        self.isPointButtonClicked = True
        self.polygonItems = []
        self.pointStack = []
        self.annotationStack = []
        self.ButtonClicked = 0
        self.currentHighlight = None
        self.currentMask = None

        # Configure list widget appearances
        self.listWidget.setMinimumWidth(250)
        self.listWidget.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.listWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Make category list match button width
        self.categoryList.setFixedWidth(200)
        self.categoryList.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Connect signals
        self.listWidget.itemClicked.connect(self.displayImageInfo)
        self.listWidget.itemClicked.connect(lambda item: self.displayImage(item, init_point=True))
        self.graphicsView.mousePressEvent = self.graphicsViewMousePressEvent
        self.annList.itemClicked.connect(self.highlightMask)

        # Initialize SAM model
        sam_checkpoint = "checkpoints/sam_vit_l_0b3195.pth"
        model_type = "vit_l"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def is_valid_image(self, filepath):
        """Check if the file is a valid image and not a hidden file."""
        # Skip hidden files and OS-specific metadata files
        filename = os.path.basename(filepath)
        if filename.startswith('.') or filename.startswith('._'):
            return False

        # Check if it's a valid image file
        try:
            with Image.open(filepath) as img:
                img.verify()
            return True
        except Exception:
            return False

    def openImageFolder(self):
        self.folderPath = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if self.folderPath:
            # Create masks folder
            self.maskFolderPath = os.path.join(os.path.dirname(self.folderPath), 'masks')
            if not os.path.exists(self.maskFolderPath):
                os.makedirs(self.maskFolderPath)

            self.listWidget.clear()

            # Filter and add only valid image files
            for file in os.listdir(self.folderPath):
                # Skip hidden files, metadata files, and files starting with '._'
                if file.startswith('.') or file.startswith('._'):
                    continue

                # Check if it's an image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        # Full path to image
                        img_path = os.path.join(self.folderPath, file)

                        # Try to open and verify the image
                        with Image.open(img_path) as img:
                            # Actually load the image data to verify it's valid
                            img.load()
                            img_data = img.getdata()  # This will fail for corrupt images

                            # If we get here, the image is valid
                            item = QListWidgetItem(file)
                            mask_filename = os.path.splitext(file)[0] + '_mask' + os.path.splitext(file)[1]
                            mask_filepath = os.path.join(self.maskFolderPath, mask_filename)
                            if os.path.exists(mask_filepath):
                                item.setForeground(QColor("red"))
                            self.listWidget.addItem(item)
                    except Exception:
                        # Skip any files that can't be opened or aren't valid images
                        continue

            # Initialize point mode if valid images were found
            if self.listWidget.count() > 0:
                self.initPointMode()

    def initPointMode(self):
        """Initialize point mode and set cursor"""
        self.isPointButtonClicked = True
        self.graphicsView.setCursor(QCursor(Qt.CrossCursor))

    def openImageFolder(self):
        self.folderPath = QFileDialog.getExistingDirectory(None, "Select Image Folder")
        if self.folderPath:
            # Create masks folder
            self.maskFolderPath = os.path.join(os.path.dirname(self.folderPath), 'masks')
            if not os.path.exists(self.maskFolderPath):
                os.makedirs(self.maskFolderPath)

            self.listWidget.clear()

            # Get all valid image files
            for file in os.listdir(self.folderPath):
                # Skip hidden files and metadata files
                if file.startswith('.') or file.startswith('._'):
                    continue

                # Check if it's an image file
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    try:
                        # Full path to image
                        img_path = os.path.join(self.folderPath, file)

                        # Try to open and verify the image
                        with Image.open(img_path) as img:
                            # Actually load the image data to verify it's valid
                            img.load()

                            # Create list item
                            item = QListWidgetItem(file)

                            # Check for corresponding mask file
                            base_name = os.path.splitext(file)[0]
                            mask_filename = f"{base_name}_mask{os.path.splitext(file)[1]}"
                            mask_filepath = os.path.join(self.maskFolderPath, mask_filename)

                            if os.path.exists(mask_filepath):
                                # Set text color to red if mask exists
                                item.setForeground(QColor("red"))

                                # If this is the first item, display it with mask
                                if self.listWidget.count() == 0:
                                    self.listWidget.addItem(item)
                                    self.displayImage(item, init_point=True)

                                    # Load and display the mask
                                    mask_img = Image.open(mask_filepath)
                                    mask_array = np.array(mask_img)
                                    # Convert to binary mask if needed
                                    if len(mask_array.shape) > 2:  # If RGB/RGBA
                                        mask_array = mask_array[:, :, 0] > 0
                                    else:
                                        mask_array = mask_array > 0
                                    self.currentMask = mask_array
                                    self.displayMask([mask_array])  # Wrap in list to match expected format
                                else:
                                    self.listWidget.addItem(item)
                            else:
                                self.listWidget.addItem(item)

                    except Exception as e:
                        print(f"Error processing image {file}: {e}")
                        continue

            # Initialize point mode if valid images were found
            if self.listWidget.count() > 0:
                self.initPointMode()

    def displayImage(self, item, init_point=False):
        """Display the selected image"""
        if not item:
            return

        image_path = os.path.join(self.folderPath, item.text())
        if not os.path.exists(image_path):
            return

        try:
            self.image = Image.open(image_path)
            pixmap = QPixmap(image_path)

            # Clear the scene and set up the new image
            self.scene.clear()

            # Create pixmap item and center it
            self.pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmapItem)

            # Center the image in the view
            self.scene.setSceneRect(self.scene.itemsBoundingRect())
            self.graphicsView.setSceneRect(self.scene.sceneRect())
            self.graphicsView.centerOn(self.pixmapItem)

            # Add mask pixmap item
            self.maskPixmapItem = QGraphicsPixmapItem()
            self.scene.addItem(self.maskPixmapItem)

            # Ensure a category is selected
            if self.categoryList.count() > 0 and not self.categoryList.currentItem():
                self.categoryList.setCurrentRow(0)

            if init_point:
                self.initPointMode()

            # Reset point stack and annotation stack
            self.pointStack.clear()
            self.annotationStack.clear()

        except Exception as e:
            print(f"Error loading image: {e}")
            return

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmapItem:
            self.graphicsView.centerOn(self.pixmapItem)

    def displayImageInfo(self, item):
        image_path = os.path.join(self.folderPath, item.text())  # Use the stored folder path
        pixmap = QPixmap(image_path)

        width = pixmap.width()
        height = pixmap.height()
        channels = pixmap.depth() // 8

        info_html = f"""
        <html>
        <head/>
        <body>
        <p><b>Image Name:</b> {item.text()}</p>
        <p><b>Width:</b> {width} pixels</p>
        <p><b>Height:</b> {height} pixels</p>
        <p><b>Channels:</b> {channels}</p>
        </body>
        </html>
        """
        self.textEdit.setHtml(info_html)

    def onPointButtonClick(self):
        self.isPointButtonClicked = True  
        self.graphicsView.setCursor(QCursor(Qt.CrossCursor)) 

    

    def graphicsViewMousePressEvent(self, event):
        if not self.isPointButtonClicked:
            QGraphicsView.mousePressEvent(self.graphicsView, event)
            return
        # Get the click position
        position = self.graphicsView.mapToScene(event.pos())

        # Check if the position is within the pixmap item
        if self.pixmapItem and self.pixmapItem.contains(position):
            # Map the position to the pixmap item's coordinate system
            relative_position = self.pixmapItem.mapFromScene(position)
            # Determine the color based on the mouse button
            if event.button() == Qt.LeftButton:
                color = Qt.red
                self.ButtonClicked = 1
            elif event.button() == Qt.RightButton:
                color = Qt.green
                self.ButtonClicked = 0
            else:
                # For other mouse buttons, you can choose to do nothing or handle them differently
                QGraphicsView.mousePressEvent(self.graphicsView, event)
                return

            # Create and add a colored dot to the scene at the click position
            dot = QGraphicsEllipseItem(relative_position.x(), relative_position.y(), 10, 10)
            dot.setBrush(QBrush(color))
            self.scene.addItem(dot)
            point_info = {
            'dot_item': dot,
            'position': [relative_position.x(), relative_position.y()],
            'label_type': self.ButtonClicked,
            'maks':[]
        }
            self.pointStack.append(point_info)
            image=np.array(self.image)
            self.predictor.set_image(image)
            mask, score, logit = self.predictor.predict(
            point_coords=np.array([point_info['position'] for point_info in self.pointStack]).astype(float),
            point_labels=np.array([point_info['label_type'] for point_info in self.pointStack]).astype(float),
            multimask_output=True,
        )
            self.pointStack[-1]['mask'] = mask
            self.displayMask(mask)
            # Optionally, you might want to store or use the click position
            print(f"Clicked position relative to image: ({relative_position.x()}, {relative_position.y()})")

        # Call the base class implementation to preserve the default behaviour
        QGraphicsView.mousePressEvent(self.graphicsView, event)
    
    def undoLastPoint(self):
        if not self.pointStack:
            return
        else: 
            last_point_info = self.pointStack.pop() 
            self.scene.removeItem(last_point_info['dot_item']) 
            
            if self.pointStack:
                mask = self.pointStack[-1]['mask']
            else:
                mask = None
                self.isPointButtonClicked = False
                self.graphicsView.setCursor(QCursor(Qt.ArrowCursor))

            self.displayMask(mask)
            
            last_position = last_point_info['position']
            print(f"Removed point at position: ({last_position})")

    def displayMask(self, mask):
        if mask is None:
            pixmap = self.pixmapItem.pixmap()
            self.scene.clear()
            self.pixmapItem = QGraphicsPixmapItem(pixmap)
            self.scene.addItem(self.pixmapItem)
            self.maskPixmapItem = QGraphicsPixmapItem()
            self.scene.addItem(self.maskPixmapItem)
        else:
            mask_image = Image.fromarray((mask[0] * 255).astype('uint8'))
            mask_image = mask_image.convert('RGBA')
            mask_data = mask_image.getdata()
            new_data = []
            for item in mask_data:
                if item[0] == 255:
                    new_data.append((255, 0, 0, 100))
                else:
                    new_data.append((255, 255, 255, 0))
            mask_image.putdata(new_data)

            qimage = ImageQt(mask_image)
            mask_pixmap = QPixmap.fromImage(qimage)

            if hasattr(self, 'maskPixmapItem'):
                self.maskPixmapItem.setPixmap(mask_pixmap)
            else:
                self.maskPixmapItem = QGraphicsPixmapItem(mask_pixmap)
                self.scene.addItem(self.maskPixmapItem)

        self.scene.update()
    
    def getMaskContourPoints(self, mask):
        binary_mask = np.uint8(mask) * 255
        assert len(mask.shape) == 2, "Mask must be a 2-dimensional array"
    
        binary_mask = np.uint8(mask) * 255
    
        assert np.setdiff1d(binary_mask, [0, 255]).size == 0, "binary_mask must contain only 0 and 255"
            
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        
        if contours:
            contour_points = largest_contour.reshape(-1, 2).tolist()
        else:
            contour_points = []

        return contour_points

    def finishAnnotation(self):
        """Finish the current annotation and save it"""
        if not self.pointStack:
            return

        # Check if a category is selected
        current_category_item = self.categoryList.currentItem()
        if not current_category_item:
            # If no category selected, select the first one
            if self.categoryList.count() > 0:
                self.categoryList.setCurrentRow(0)
                current_category_item = self.categoryList.currentItem()
            else:
                # If no categories exist, add default category
                self.categoryList.addItem("1")
                self.categoryList.setCurrentRow(0)
                current_category_item = self.categoryList.currentItem()

        # Get the current category
        category = current_category_item.text()

        # Get the last annotation info
        last_annotation_info = self.pointStack[-1]
        self.currentMask = last_annotation_info['mask'][0]  # Store the current mask

        # Clear points and update display
        self.pointStack.clear()
        self.displayMask(None)

        # Save the mask
        self.saveMaskToFile()

    def saveMaskToFile(self):
        if hasattr(self, 'image') and self.image is not None and self.currentMask is not None:
            # Get the original image filename and extension
            image_filename = os.path.basename(self.image.filename)
            base_name, ext = os.path.splitext(image_filename)

            # Create mask filename with same extension as original
            mask_filename = f"{base_name}_mask{ext}"
            mask_path = os.path.join(self.maskFolderPath, mask_filename)

            # Convert mask to binary image (255 for white, 0 for black)
            binary_mask = (self.currentMask * 255).astype(np.uint8)

            # Save the mask using the same format as the original image
            mask_image = Image.fromarray(binary_mask)
            # Convert to RGB if the format requires it (e.g., JPEG)
            if ext.lower() in ['.jpg', '.jpeg']:
                mask_image = mask_image.convert('RGB')
            mask_image.save(mask_path, quality=100)  # Use high quality for JPEG

            # Update the list widget item color to indicate annotation
            current_item = self.listWidget.findItems(image_filename, Qt.MatchExactly)[0]
            current_item.setForeground(QColor("red"))

            print(f"Mask saved to: {mask_path}")

    def drawPolygon(self):
        for annotation_info in self.annotationStack:
            points = annotation_info['points']
            
            polygon = QPolygonF()
            for point in points:
                polygon.append(QPointF(point[0], point[1]))

            polygon_item = QGraphicsPolygonItem(polygon)

            polygon_item.setPen(QPen(Qt.black, 2))
            
            polygon_item.setBrush(QBrush(QColor(255, 0, 0, 100)))

            self.scene.addItem(polygon_item)
    
    def updateAnnList(self):
        self.annList.clear()  

        for annotation in self.annotationStack:
            label = annotation['label']
            item = QListWidgetItem(label) 
            self.annList.addItem(item) 

    def deleteAnnotation(self):
        currentRow = self.annList.currentRow() 

        if currentRow != -1:
            del self.annotationStack[currentRow] 
            self.annList.takeItem(currentRow)  
            self.drawPolygon()
            self.updateAnnList()

    def deleteAllPoints(self):
        """Remove all points and masks from the current image"""
        # Clear all points from the scene
        for point_info in self.pointStack:
            if point_info['dot_item'] in self.scene.items():
                self.scene.removeItem(point_info['dot_item'])

        # Clear the point stack
        self.pointStack.clear()

        # Clear any displayed mask
        self.displayMask(None)

        # Reset current mask
        self.currentMask = None

        # Update the scene
        self.scene.update()

        # Reset point mode
        self.isPointButtonClicked = True
        self.graphicsView.setCursor(QCursor(Qt.CrossCursor))

    def highlightMask(self):
        currentRow = self.annList.currentRow()  
        annotation_info = self.annotationStack[currentRow] 
        mask_contour_points = annotation_info['points']  

        if self.currentHighlight:
            if self.currentHighlight.scene() == self.scene:
                self.scene.removeItem(self.currentHighlight)
            self.currentHighlight = None

        polygon = QPolygonF()
        for point in mask_contour_points:
            polygon.append(QPointF(point[0], point[1]))

        polygon_item = QGraphicsPolygonItem(polygon)

        polygon_item.setPen(QPen(Qt.red, 4))  
        polygon_item.setBrush(QBrush(Qt.yellow)) 


        self.scene.addItem(polygon_item)
        self.currentHighlight = polygon_item

    def saveAnnotationsToFile(self):
        if hasattr(self, 'image') and self.image is not None:
            image_path = self.image.filename
            json_path = image_path.rsplit('.', 1)[0] + '.json'
            
            json_data = {
                "version": "3.16.7",
                "flags": {},
                "shapes": self.annotationStack,
                "lineColor": [0, 255, 0, 128],
                "fillColor": [255, 0, 0, 128],
                "imagePath": os.path.basename(image_path),
                "imageData": None,
                "imageHeight": self.image.height,
                "imageWidth": self.image.width
            }
            
            dir_name = os.path.dirname(json_path)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=4)


            

