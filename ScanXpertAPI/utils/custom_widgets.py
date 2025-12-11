# custom_widgets.py
import pandas as pd
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.metrics import dp
from kivy.uix.label import Label
from kivy.properties import StringProperty, NumericProperty
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.slider import Slider
import math
from kivy.app import App
from kivy.core.window import Window
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


def calculate_pixel_spacing(depth, fov, type, num_horizontal_pixels=512, num_vertical_pixels=512):
    """Calculates the pixel spacing with respect to the image selected and from the transducer specification

    Args:
        depth (float): The axial depth of the image being visualized
        fov (int(degrees)): Transducer field of view
        type (string): Either linear or convex probe
        num_horizontal_pixels (int, optional): Image size. Defaults to 512.
        num_vertical_pixels (int, optional): Image size. Defaults to 512.

    Returns:
        float: pixel spacing in cm/px
    """
    depth = depth * 10 # cm to mm
    if type == "Convex":
        fov_radians = math.radians(fov)
        horizontal_pixel_spacing = (depth * fov_radians) / num_horizontal_pixels
    else:
        horizontal_pixel_spacing = fov / num_horizontal_pixels
    vertical_pixel_spacing = depth / num_vertical_pixels
    combined_pixel_spacing = math.sqrt((horizontal_pixel_spacing**2 + vertical_pixel_spacing**2) / math.sqrt(2))
    return combined_pixel_spacing / 10 # in cm

class CustomDropDown(DropDown):
    def __init__(self, items, **kwargs):
        super(CustomDropDown, self).__init__(**kwargs)
        """Creates a dropdown button with the list items as provided
        """
        self.scroll_view = ScrollView(size_hint=(1, None), height=Window.height * 0.3)
        self.layout = BoxLayout(orientation='vertical', size_hint_y=None)
        self.layout.bind(minimum_height=self.layout.setter('height'))
        
        for item in items:
            btn = Button(text=str(item), size_hint_y=None, height=35, 
                         background_color=(0.827, 0.827, 0.827, 1), color=(1, 1, 1, 1))
            btn.bind(on_release=lambda btn: self.select(btn.text))
            self.layout.add_widget(btn)
        
        self.scroll_view.add_widget(self.layout)
        self.add_widget(self.scroll_view)

class TopBar(BoxLayout):
    def __init__(self, **kwargs):
        super(TopBar, self).__init__(**kwargs)
        self.app = App.get_running_app()
        self.orientation = 'vertical'
        self.spacing = '10dp'
        self.padding = '10dp'

        # Sample DataFrame - replace with your actual data
        self.df = pd.read_excel(resource_path(r"utils/transducer_model.xlsx"))
        # Dropdown
        dropdown = CustomDropDown(items=self.df['MODEL'])
        self.main_button = Button(text=self.app.selected_item, size_hint=(0.6, None),height=dp(40),
                                  background_color=(0.2, 0.6, 1, 1), color=(1, 1, 1, 1), pos_hint={"center_x": 0.5})
        self.main_button.bind(on_release=dropdown.open)
        dropdown.bind(on_select=self.on_dropdown_select)
        self.add_widget(self.main_button)
        
        # Depth input
        depth_layout = BoxLayout(orientation='horizontal', size_hint=(0.8, None), height=dp(40), pos_hint={"center_x": 0.5, "center_y": 0.6})
        label = Label(text="Depth (cm)", size_hint=(0.5, 1), color=(1, 1, 1, 1))
        self.depth_slider = Slider(min=4.0, max=20.0, step = 0.1, value = self.app.depth, size_hint=(None, None), size=(200, 40), pos_hint={"center_x": 0.5, "center_y": 0.6})
        self.depth_input = Label(text=f"{self.depth_slider.value:.1f}", size_hint=(0.5, 1), color=(1, 1, 1, 1))
        self.depth_slider.bind(value=self.on_depth_change)
        depth_layout.add_widget(label)
        depth_layout.add_widget(self.depth_input)
        self.add_widget(depth_layout)
        self.add_widget(self.depth_slider)
        
    def on_dropdown_select(self, instance, value):
        """
        Handles the selection of an item from the dropdown menu.
        
        Parameters:
        instance (CustomDropDown): The dropdown instance triggering the event.
        value (str): The selected item from the dropdown.

        Updates the selected transducer model in the application and triggers pixel spacing recalculation.
        """
        self.app.selected_item = value
        self.main_button.text = value
        self.update_pixel_spacing()
    
    def on_depth_change(self, instance, value):
        """
        Handles changes in the depth slider value.
        
        Parameters:
        instance (Slider): The slider instance triggering the event.
        value (float): The new depth value selected by the user.

        Updates the application's depth value, updates the depth input label, and recalculates pixel spacing.
        """
        try:
            self.app.depth = round(value, 1)
            self.depth_input.text = f"{value:.1f}"
            self.update_pixel_spacing()
        except ValueError:
            pass
    
    def update_pixel_spacing(self):
        """
        Recalculates and updates the pixel spacing value based on the selected transducer model and depth.
        
        Retrieves relevant parameters such as Field of View (FOV) and transducer type from the loaded dataset.
        Fetches the image texture size (if available) to adjust calculations.
        Updates the pixel spacing input field with the calculated value.
        
        Handles exceptions and logs errors if pixel spacing calculation fails.
        """
        if self.app.selected_item != "Select Transducer Model":
            try:
                fov = self.df.loc[self.df['MODEL'] == self.app.selected_item, 'FOV'].values[0]
                type = self.df.loc[self.df['MODEL'] == self.app.selected_item, 'TYPE'].values[0]
                app = App.get_running_app()
                measurement_screen = app.root.ids.content_manager.get_screen("Measurements").ids.measurements_manager.get_screen("Auto")
                image = measurement_screen.ids.image_for_measurement
                self.pixel_spacing_input = measurement_screen.ids.pixel
                
                if image.texture:
                    w, h = image.texture.size
                    print(w, h)
                    self.pixel_spacing = calculate_pixel_spacing(self.app.depth, fov, type, w, h)
                else:
                    self.pixel_spacing = calculate_pixel_spacing(self.app.depth, fov, type)
                
                self.pixel_spacing_input.text = f"{self.pixel_spacing:.6f}"
            except Exception as e:
                print(f"Error updating pixel spacing: {e}")
                self.pixel_spacing_input.text = "Error"

