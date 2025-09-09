import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSlider, QPushButton, QLabel, QGridLayout,
                             QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal

class InputPanel(QMainWindow):
    # Signal to emit when values change
    values_changed = pyqtSignal(dict)
    
    def __init__(self, title="Control Panel", width=400, height=600):
        # Ensure there's a QApplication instance
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)
        
        super().__init__()
        
        # Initialize UI
        self.setWindowTitle(title)
        self.setGeometry(100, 100, width, height)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create a grid layout for controls
        controls_layout = QGridLayout()
        main_layout.addLayout(controls_layout)
        
        # Initialize value dictionary
        self.values = {}
        
        # Create sliders
        self.sliders = {}
        slider_names = ["Intensity", "Speed","Sensor", "Danger","Joyful","Sad","Angry","Curious","Passionate","Rage","Contemplative","Neutral"]
        
        
        for i, name in enumerate(slider_names):
            # Create a horizontal layout for each slider
            row_layout = QHBoxLayout()
            
            # Add label
            label = QLabel(name)
            label.setMinimumWidth(80)
            row_layout.addWidget(label)
            
            # Create slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(10)
            
            # Create value label
            value_label = QLabel("0")
            value_label.setMinimumWidth(30)
            
            # Connect slider value change to update function
            slider_id = name.lower().replace(' ', '_')
            self.values[slider_id] = 0
            slider.valueChanged.connect(lambda val, s_id=slider_id, vl=value_label: self.update_slider_value(val, s_id, vl))
            
            row_layout.addWidget(slider)
            row_layout.addWidget(value_label)
            
            # Add to grid layout
            controls_layout.addLayout(row_layout, i, 0)
            
            # Store slider reference
            self.sliders[slider_id] = slider
        
        # Add some spacing
        main_layout.addSpacing(20)
        
        # Create mode buttons section
        main_layout.addWidget(QLabel("Modes:"))
        mode_buttons_layout = QGridLayout()
        main_layout.addLayout(mode_buttons_layout)
        
        # Create mode button group for exclusive selection
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)  # Make buttons exclusive
        
        mode_names = ["Awaken", "Inactive", "Mode 3", "Mode 4"]
        
        # Default selected mode
        self.values['selected_mode'] = 'Waiting'
        
        self.mode_buttons = {}
        for i, name in enumerate(mode_names):
            button = QPushButton(name)
            button.setCheckable(True)  # Make buttons checkable
            
            button_id = name.lower().replace(' ', '_')
            
            # Add to button group
            self.mode_group.addButton(button, i)
            
            # Connect button to handler
            button.clicked.connect(lambda checked, b_id=button_id: self.update_mode_selection(b_id))
            
            # Add to grid layout (2 columns of buttons)
            row = i // 2
            col = i % 2
            mode_buttons_layout.addWidget(button, row, col)
            
            # Store button reference
            self.mode_buttons[button_id] = button
        
        # Add effect buttons section
        main_layout.addSpacing(10)
        main_layout.addWidget(QLabel("Effects:"))
        effect_buttons_layout = QGridLayout()
        main_layout.addLayout(effect_buttons_layout)
        
        # Create effect button group for exclusive selection
        self.effect_group = QButtonGroup(self)
        self.effect_group.setExclusive(True)  # Make buttons exclusive
        
        effect_names = ["Effect 1", "Effect 2", "Effect 3", "Effect 4"]
        
        # Default selected effect
        self.values['selected_effect'] = 'none'
        
        self.effect_buttons = {}
        for i, name in enumerate(effect_names):
            button = QPushButton(name)
            button.setCheckable(True)  # Make buttons checkable
            
            button_id = name.lower().replace(' ', '_')
            
            # Add to button group
            self.effect_group.addButton(button, i)
            
            # Connect button to handler
            button.clicked.connect(lambda checked, b_id=button_id: self.update_effect_selection(b_id))
            
            # Add to grid layout (2 columns of buttons)
            row = i // 2
            col = i % 2
            effect_buttons_layout.addWidget(button, row, col)
            
            # Store button reference
            self.effect_buttons[button_id] = button
        
        # Add reset button at the bottom
        main_layout.addSpacing(20)
        reset_button = QPushButton("Reset All")
        reset_button.clicked.connect(self.reset_values)
        main_layout.addWidget(reset_button)
        
        # Show the window
        self.show()
        
    def update_slider_value(self, value, slider_id, value_label):
        """Update the slider value in the dictionary and update the label"""
        self.values[slider_id] = value
        value_label.setText(str(value))
        self.values_changed.emit(self.values)
        
    def update_mode_selection(self, button_id):
        """Update selected mode"""
        # Set the selected mode
        self.values['selected_mode'] = button_id
        
        # Set all mode buttons to 0 and the selected one to 1
        for mode_id, button in self.mode_buttons.items():
            # Create a numeric value for each mode (0 = off, 1 = on)
            mode_value_key = f"mode_{mode_id}"
            if mode_id == button_id and button.isChecked():
                self.values[mode_value_key] = 1
                button.setStyleSheet("background-color: #8CC84B;")
            else:
                self.values[mode_value_key] = 0
                button.setStyleSheet("")
                
        self.values_changed.emit(self.values)
        
    def update_effect_selection(self, button_id):
        """Update selected effect"""
        # Set the selected effect
        self.values['selected_effect'] = button_id
        
        # Set all effect buttons to 0 and the selected one to 1
        for effect_id, button in self.effect_buttons.items():
            # Create a numeric value for each effect (0 = off, 1 = on)
            effect_value_key = f"effect_{effect_id}"
            if effect_id == button_id and button.isChecked():
                self.values[effect_value_key] = 1
                button.setStyleSheet("background-color: #8CC84B;")
            else:
                self.values[effect_value_key] = 0
                button.setStyleSheet("")
                
        self.values_changed.emit(self.values)
        
    def reset_values(self):
        """Reset all controls to default values"""
        # Reset sliders
        for slider_id, slider in self.sliders.items():
            slider.setValue(50)
            
        # Reset mode buttons
        self.mode_group.setExclusive(False)  # Temporarily disable exclusivity
        for button_id, button in self.mode_buttons.items():
            button.setChecked(False)
            button.setStyleSheet("")
            self.values[f"mode_{button_id}"] = 0
        self.mode_group.setExclusive(True)  # Re-enable exclusivity
        self.values['selected_mode'] = 'none'
            
        # Reset effect buttons
        self.effect_group.setExclusive(False)  # Temporarily disable exclusivity
        for button_id, button in self.effect_buttons.items():
            button.setChecked(False)
            button.setStyleSheet("")
            self.values[f"effect_{button_id}"] = 0
        self.effect_group.setExclusive(True)  # Re-enable exclusivity
        self.values['selected_effect'] = 'none'
            
        self.values_changed.emit(self.values)
    
    def get_values(self):
        """Return the current values dictionary"""
        return self.values.copy()

# For testing the panel independently
if __name__ == "__main__":
    def print_values(values):
        print("Values updated:", values)
    
    app = QApplication(sys.argv)
    panel = InputPanel()
    panel.values_changed.connect(print_values)
    
    # Example of getting values directly
    print("Initial values:", panel.get_values())
    
    sys.exit(app.exec_())