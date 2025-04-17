# ai_mcq_processor.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class AIMCQProcessor:
    def __init__(self):
   
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a simple CNN for bubble detection"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='softmax')  # 2 classes: filled or unfilled
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_bubble_sample(self, img):
        """Convert a bubble ROI to model input format"""
   
        sample = cv2.resize(img, (48, 48))
        
     
        if len(sample.shape) > 2 and sample.shape[2] > 1:
            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        
      
        sample = sample / 255.0
      
        sample = sample.reshape(1, 48, 48, 1)
        
        return sample
    
    def is_bubble_filled(self, bubble_img):
        """Use CNN to determine if a bubble is filled"""
       
        gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY) if len(bubble_img.shape) > 2 else bubble_img
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        fill_ratio = cv2.countNonZero(thresh) / thresh.size
        
        return fill_ratio > 0.3  # If more than 30% is filled
    
    def segment_answer_sheet(self, img):
        """Segment the answer sheet to identify the grid structure"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours of the grid
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour (likely the answer grid)
        grid_contour = max(contours, key=cv2.contourArea) if contours else None
        
        return grid_contour, thresh
    
    def detect_bubbles(self, img, thresh, grid_contour=None):
        """Detect individual bubble areas"""
        if grid_contour is not None:
            # Create a mask for the grid area
            mask = np.zeros_like(thresh)
            cv2.drawContours(mask, [grid_contour], 0, 255, -1)
            
            # Apply mask to threshold image
            thresh = cv2.bitwise_and(thresh, mask)
        
        # Find contours of potential bubbles
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to identify bubble shapes
        bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 500:  # Adjust based on your answer sheet
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.6:  # More circular
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        if 0.8 < aspect_ratio < 1.2:  # Near square/circle
                            center_x, center_y = x + w//2, y + h//2
                            bubbles.append((center_x, center_y, contour))
        
        return bubbles
    
    def organize_bubbles_into_grid(self, bubbles, num_questions, options_per_row):
        """Organize detected bubbles into a logical grid structure"""
        # Sort bubbles by y-coordinate (row)
        bubbles.sort(key=lambda b: b[1])
        
    
        y_coords = [b[1] for b in bubbles]
        y_coords.sort()
        
        # Calculate differences between consecutive y-coordinates
        y_diffs = [y_coords[i+1] - y_coords[i] for i in range(len(y_coords)-1)]
        
        # Find the average gap that represents the spacing between rows
        if y_diffs:
    
            threshold = np.mean(y_diffs) / 2
            row_gaps = [diff for diff in y_diffs if diff > threshold]
            if row_gaps:
                avg_row_gap = np.mean(row_gaps)
            else:
                avg_row_gap = np.mean(y_diffs)
        else:
            
            avg_row_gap = 20  # Default value
        
        # Group bubbles into rows
        rows = []
        current_row = []
        
        for i, bubble in enumerate(bubbles):
            if i == 0:
                current_row.append(bubble)
            else:
              
                if abs(bubble[1] - bubbles[i-1][1]) < avg_row_gap/2:
                    current_row.append(bubble)
                else:
                  
                    current_row.sort(key=lambda b: b[0])
                    rows.append(current_row)
                    current_row = [bubble]
        
      
        if current_row:
            current_row.sort(key=lambda b: b[0])
            rows.append(current_row)
        
      
        rows = rows[:num_questions]
        
        return rows
    
    def process_answer_sheet(self, image_path, num_questions, options_per_question):
        """Process an MCQ answer sheet and extract marked answers"""
        # Load and preprocess the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = img.shape[:2]
        max_dimension = 1200
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        grid_contour, thresh = self.segment_answer_sheet(img)
        
        
        bubbles = self.detect_bubbles(img, thresh, grid_contour)
        
        
        rows = self.organize_bubbles_into_grid(bubbles, num_questions, options_per_question)
        
        # Process each row to find marked bubbles
        answers = {}
        options = ['A', 'B', 'C', 'D', 'E'][:options_per_question]
        
        for q_idx, row in enumerate(rows, start=1):
            if len(row) < 2:  # Skip rows with too few bubbles detected
                continue
                
            # Ensure we limit to the expected number of options
            row = row[:options_per_question]
            
            # Check each bubble to see if it's filled
            filled = []
            for op_idx, (center_x, center_y, contour) in enumerate(row):
                # Extract region around the bubble
                x, y, w, h = cv2.boundingRect(contour)
                # Add some margin
                margin = max(w, h) // 4
                x_min = max(0, x - margin)
                y_min = max(0, y - margin)
                x_max = min(img.shape[1], x + w + margin)
                y_max = min(img.shape[0], y + h + margin)
                
                bubble_img = img[y_min:y_max, x_min:x_max]
                
                # Check if bubble is filled
                if self.is_bubble_filled(bubble_img):
                    filled.append((op_idx, 1.0))  # 1.0 represents confidence
                else:
                    filled.append((op_idx, 0.0))
            
            # Find the most filled bubble
            if filled:
                most_filled = max(filled, key=lambda x: x[1])
                if most_filled[1] > 0:
                    answers[q_idx] = options[most_filled[0]]
        
        # Fill in any missing questions
        for q_idx in range(1, num_questions + 1):
            if q_idx not in answers:
                answers[q_idx] = options[0]  # Default to first option if not detected
        
        return answers
    
    def visualize_detection(self, image_path, bubbles=None, answers=None):
        """Create a visualization of the detection process"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Create a copy for marking
        vis_img = img.copy()
        
        # If bubbles were detected, draw them
        if bubbles:
            for center_x, center_y, contour in bubbles:
                # Draw contour
                cv2.drawContours(vis_img, [contour], -1, (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(vis_img, (center_x, center_y), 2, (0, 0, 255), -1)
        
        # If answers were detected, label them
        if answers:
            font = cv2.FONT_HERSHEY_SIMPLEX
            for q_idx, answer in answers.items():
                # Position for text
                text_x = img.shape[1] - 150
                text_y = (q_idx * img.shape[0]) // (len(answers) + 1)
                
                # Draw labeled answer
                cv2.putText(
                    vis_img,
                    f"Q{q_idx}: {answer}",
                    (text_x, text_y),
                    font,
                    0.7,
                    (255, 0, 0),
                    2
                )
        
        # Save the visualization
        output_path = image_path.replace(".jpg", "_viz.jpg").replace(".png", "_viz.png")
        cv2.imwrite(output_path, vis_img)
        
        return output_path