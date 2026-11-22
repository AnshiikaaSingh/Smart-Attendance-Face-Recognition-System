import cv2
import mediapipe as mp
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial import distance as dist

class AttentivenessDetector:
    """Detect student attentiveness based on eye aspect ratio (EAR)"""
    
    def __init__(self, ear_threshold=0.25, consecutive_frames=2):
        """
        Initialize attentiveness detector
        
        Args:
            ear_threshold: Eye Aspect Ratio threshold (below = closed/drowsy)
            consecutive_frames: Number of consecutive frames to confirm drowsiness
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        # MediaPipe Face Mesh for eye landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe Face Mesh
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 362, 385, 387, 263, 373, 380
        self.LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    
    def calculate_ear(self, eye_landmarks: np.ndarray) -> float:
        """
        Calculate Eye Aspect Ratio (EAR)
        
        Formula: EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        where p1-p6 are the 6 facial landmarks of an eye
        
        Args:
            eye_landmarks: Array of 6 (x, y) coordinates
            
        Returns:
            Eye aspect ratio value
        """
        # Compute vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def get_eye_landmarks(self, face_landmarks, image_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract eye landmarks from face mesh
        
        Args:
            face_landmarks: MediaPipe face landmarks
            image_shape: (height, width) of the image
            
        Returns:
            Tuple of (left_eye_landmarks, right_eye_landmarks)
        """
        h, w = image_shape
        
        # Extract left eye landmarks
        left_eye = []
        for idx in self.LEFT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            left_eye.append([landmark.x * w, landmark.y * h])
        
        # Extract right eye landmarks
        right_eye = []
        for idx in self.RIGHT_EYE_INDICES:
            landmark = face_landmarks.landmark[idx]
            right_eye.append([landmark.x * w, landmark.y * h])
        
        return np.array(left_eye), np.array(right_eye)
    
    def detect_attentiveness(self, face_image: np.ndarray) -> Dict[str, any]:
        """
        Detect if a person is attentive based on eye state
        
        Args:
            face_image: Cropped face image (BGR format)
            
        Returns:
            Dictionary with attentiveness information
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_image.shape
        
        # Process with Face Mesh
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return {
                'is_attentive': True,  # Default to attentive if can't detect
                'eye_state': 'unknown',
                'left_ear': None,
                'right_ear': None,
                'avg_ear': None
            }
        
        # Get first face (should be only one in cropped image)
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract eye landmarks
        left_eye, right_eye = self.get_eye_landmarks(face_landmarks, (h, w))
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Determine eye state
        if avg_ear < self.ear_threshold * 0.7:
            eye_state = 'closed'
            is_attentive = False
        elif avg_ear < self.ear_threshold:
            eye_state = 'semi-closed'
            is_attentive = False
        else:
            eye_state = 'open'
            is_attentive = True
        
        return {
            'is_attentive': is_attentive,
            'eye_state': eye_state,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear
        }
    
    def batch_analyze_attentiveness(self, recognized_faces: Dict[str, List[Dict]], 
                                   detected_faces: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """
        Analyze attentiveness for all recognized faces across timestamps
        
        Args:
            recognized_faces: Dictionary with timestamps and recognized face info
            detected_faces: Dictionary with timestamps and detected face data (contains face_image)
            
        Returns:
            Dictionary with student names and their attentiveness statistics
        """
        attentiveness_data = {}
        
        for timestamp in recognized_faces.keys():
            recognized_list = recognized_faces[timestamp]
            detected_list = detected_faces[timestamp]
            
            for rec_face, det_face in zip(recognized_list, detected_list):
                name = rec_face['name']
                
                if name == 'Unknown':
                    continue
                
                # Initialize student data if not exists
                if name not in attentiveness_data:
                    attentiveness_data[name] = {
                        'attentive_count': 0,
                        'inattentive_count': 0,
                        'total_snapshots': 0,
                        'eye_states': []
                    }
                
                # Detect attentiveness
                attention_result = self.detect_attentiveness(det_face['face_image'])
                
                # Update statistics
                attentiveness_data[name]['total_snapshots'] += 1
                if attention_result['is_attentive']:
                    attentiveness_data[name]['attentive_count'] += 1
                else:
                    attentiveness_data[name]['inattentive_count'] += 1
                
                attentiveness_data[name]['eye_states'].append({
                    'timestamp': timestamp,
                    'state': attention_result['eye_state'],
                    'ear': attention_result['avg_ear']
                })
        
        return attentiveness_data
    
    def __del__(self):
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
