import numpy as np
import cv2
import mediapipe as mp

class BowlingActionAnalyser: 
    def __init__(self, video = None, bowler ='right arm'):
        self.video = video
        self.bowler = bowler
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose 
    
    def calculate_angle(self, a,b,c):
        a = np.array(a)
        b = np.array(b) 
        c = np.array(c) 
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle >180.0:
            angle = 360-angle
        return angle 
    
    def analyze(self): 
        state = 'OK'
        if self.video == None:
            pos = 0
        else:
            pos = self.video

        cap = cv2.VideoCapture(pos)   
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read() 
                if ret:
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False 
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
                    try:
                        landmarks = results.pose_landmarks.landmark 

                        left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        left_elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        left_wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                        right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                        left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        
                        l_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                        r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                        right_leg = self.calculate_angle(right_hip, right_knee, right_ankle)
                        left_leg = self.calculate_angle(left_hip, left_knee, left_ankle)
                        height, width, channels = image.shape

                        cv2.putText(image, str(l_angle), 
                                    tuple(np.multiply(left_elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(r_angle), 
                                    tuple(np.multiply(r_elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )      

                        cv2.putText(image, str(right_leg), 
                                    tuple(np.multiply(right_knee, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )   
                        
                        cv2.putText(image, str(left_leg), 
                                    tuple(np.multiply(left_knee, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )        
                        
                        ## Main Logic Of the Ilegal Action Detector ##
                        if (self.bowler == 'right arm'):
                            if (right_leg <160) and (left_leg >170):
                                if (r_angle<150):
                                    state = 'Not Allowed'
                                else:
                                    state = 'OK'
                        elif (self.bowler == 'left arm'):
                            if (left_leg <160) and (right_leg >170):
                                if (l_angle<150):
                                    state = 'Not Allowed'
                                else:
                                    state = 'OK'
                    except:
                        pass
                    
                    cv2.rectangle(image, (0, 0), (325, 73), (245, 117, 16), -1)  
                    cv2.rectangle(image, (430, 0), (250, 73), (245, 117, 16), -1)
                    
                    cv2.putText(image, 'STATE', (25,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, state, 
                                (20,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )               
                    
                    cv2.imshow('Mediapipe Feed', image)

                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()
class GymExcerciseAnalyzer:
    def __init__(self, excercise, video = None):
        self.video = video 
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose 
        self.excercise = excercise
    
    def calculate_angle(self, a,b,c):
        a = np.array(a)
        b = np.array(b) 
        c = np.array(c) 
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        if angle >180.0:
            angle = 360-angle
        return angle 
    
    def analyze(self): 
        if self.video == None:
            pos = 0
        else:
            pos = self.video

        cap = cv2.VideoCapture(pos) 
        counter = 0 
        stage = None
 
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read() 
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False 
                results = pose.process(image) 
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
               
                height, width, channels = image.shape

                if (self.excercise=="Push-Ups"):    
                    try:
                        landmarks = results.pose_landmarks.landmark 
                        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angle = self.calculate_angle(shoulder, elbow, wrist)
                        r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)    
                        cv2.putText(image, str(angle), 
                                    tuple(np.multiply(elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(r_angle), 
                                    tuple(np.multiply(r_elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )           
                        
                        if (angle>160 and r_angle>160):
                            stage = "up"
                        if (angle < 120 and r_angle < 120) and stage =='up':
                            stage="down"
                            counter +=1
                            print(counter) 
                    except:
                        pass
                     
                    cv2.rectangle(image, (0, 0), (325, 73), (245, 117, 16), -1) 
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    cv2.rectangle(image, (130, 0), (250, 73), (245, 117, 16), -1)
                     
                    cv2.putText(image, 'STAGE', (95,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (90,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                     
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )               
                    
                    cv2.imshow('Mediapipe Feed', image) 

                elif (self.excercise=="Curls"): 
                    try:
                        landmarks = results.pose_landmarks.landmark 
                        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        r_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        r_elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                        r_wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                        
                        angle = self.calculate_angle(shoulder, elbow, wrist)
                        r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist) 
                        angle = self.calculate_angle(shoulder, elbow, wrist)
                        r_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
                        cv2.putText(image, str(angle), 
                                    tuple(np.multiply(elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )

                        cv2.putText(image, str(r_angle), 
                                    tuple(np.multiply(r_elbow, [width, height]).astype(int)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                            )        
                        if angle > 160 and r_angle > 160:
                            stage = "down"
                        if (angle < 30 or r_angle < 30) and stage =='down':
                            stage="up"
                            counter +=1 
                    except:
                        pass
                                 
                    cv2.rectangle(image, (0, 0), (325, 73), (245, 117, 16), -1) 
                    cv2.putText(image, 'REPS', (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(counter), 
                                (10,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
                    
                    cv2.rectangle(image, (130, 0), (250, 73), (245, 117, 16), -1) 
                    cv2.putText(image, 'STAGE', (95,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(image, stage, 
                                (90,60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA) 
                    
                    self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                            )    
                    cv2.imshow('Mediapipe Feed', image)           
                else:
                    print("Invalid Excercise Name. Please Enter 'Push-Ups' or 'Curls' as excercise name.")
                    break

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
