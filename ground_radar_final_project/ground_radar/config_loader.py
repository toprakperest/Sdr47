# ground_radar/config_loader.py
import json
import os
import yaml

class ConfigLoader:
    @staticmethod
    def load_settings(settings_file="settings.yaml"):
        """
        Loads settings from a YAML file.
        Args:
            settings_file (str): Path to the settings file.
        Returns:
            dict: Loaded settings.
        """
        settings_path = os.path.join(os.path.dirname(__file__), settings_file)
        if not os.path.exists(settings_path):
            print(f"Settings file not found: {settings_path}")
            return {}

        with open(settings_path, 'r', encoding='utf-8') as file:

            settings = yaml.safe_load(file)
        return settings

    @staticmethod
    def load_freq_profiles(freq_profiles_file="freq_profiles.json"):
        """
        Loads frequency profiles from a JSON file.
        Args:
            freq_profiles_file (str): Path to the frequency profiles file.
        Returns:
            dict: Loaded frequency profiles.
        """
        freq_profiles_path = os.path.join(os.path.dirname(__file__), freq_profiles_file)
        if not os.path.exists(freq_profiles_path):
            print(f"Frequency profiles file not found: {freq_profiles_path}")
            return {}

        with open(freq_profiles_path, 'r', encoding='utf-8') as file:
            freq_profiles = json.load(file)
        return freq_profiles