import numpy as np
from sklearn.ensemble import IsolationForest
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window

# Set background to a clean clinical white/grey
Window.clearcolor = (0.95, 0.95, 0.95, 1)

class VitalsAnomalyDetector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=30, spacing=15, **kwargs)
        
        # 1. Initialize and Train the REAL AI Model on Startup
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.train_ai_model()
        
        # 2. Build the UI
        self.add_widget(Label(
            text="Real-Time AI Anomaly Detector", 
            font_size=24, bold=True, color=(0.1, 0.1, 0.3, 1), size_hint=(1, 0.2)
        ))
        
        self.hr_input = TextInput(hint_text="Heart Rate (e.g., 75)", input_filter='float', font_size=20, multiline=False)
        self.spo2_input = TextInput(hint_text="Blood Oxygen SpO2% (e.g., 98)", input_filter='float', font_size=20, multiline=False)
        
        self.add_widget(self.hr_input)
        self.add_widget(self.spo2_input)
        
        self.scan_btn = Button(
            text="Run ML Anomaly Scan", 
            size_hint=(1, 0.3), 
            background_color=(0.2, 0.4, 0.8, 1),
            font_size=20, bold=True
        )
        self.scan_btn.bind(on_press=self.detect_anomaly)
        self.add_widget(self.scan_btn)
        
        self.result_label = Label(
            text="Model Trained. Ready for patient data.", 
            font_size=18, bold=True, color=(0.3, 0.3, 0.3, 1), size_hint=(1, 0.3)
        )
        self.add_widget(self.result_label)

    def train_ai_model(self):
        """
        Generates synthetic healthy baseline data and trains the ML model.
        Healthy HR: ~60-100 bpm. Healthy SpO2: ~95-100%.
        """
        np.random.seed(42)
        # Generate 1000 data points of normal patient vitals
        normal_hr = np.random.normal(loc=80, scale=10, size=1000)
        normal_spo2 = np.random.normal(loc=98, scale=1, size=1000)
        
        # Keep realistic bounds
        normal_hr = np.clip(normal_hr, 60, 100)
        normal_spo2 = np.clip(normal_spo2, 95, 100)
        
        # Combine into features (X) and fit the Isolation Forest
        X_train = np.column_stack((normal_hr, normal_spo2))
        self.model.fit(X_train)
        print("ML Model Successfully Trained!")

    def detect_anomaly(self, instance):
        try:
            # Grab user inputs
            hr = float(self.hr_input.text)
            spo2 = float(self.spo2_input.text)
            
            # Format data for Scikit-Learn
            new_patient_data = np.array([[hr, spo2]])
            
            # Use the trained AI model to predict
            # IsolationForest returns 1 for normal (inliers) and -1 for anomalies (outliers)
            prediction = self.model.predict(new_patient_data)
            
            if prediction[0] == 1:
                self.result_label.text = f"Result: NORMAL\nHR: {hr}, SpO2: {spo2} fit the healthy baseline."
                self.result_label.color = (0.1, 0.6, 0.1, 1) # Green
            else:
                self.result_label.text = f"CRITICAL ANOMALY DETECTED!\nVitals are outside safe ML boundaries."
                self.result_label.color = (0.8, 0.1, 0.1, 1) # Red
                
        except ValueError:
            self.result_label.text = "Error: Please enter valid numbers."
            self.result_label.color = (0.8, 0.5, 0.1, 1) # Orange

class RealHealthAIApp(App):
    def build(self):
        self.title = 'AI Anomaly Detector'
        return VitalsAnomalyDetector()

if __name__ == '__main__':
    RealHealthAIApp().run()
