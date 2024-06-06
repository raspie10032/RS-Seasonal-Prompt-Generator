import random
import csv
import os

# Constants for file paths
DATA_DIR = 'data'
FILES = {
    'seasonal_fashion': 'seasonal_fashion.csv',
    'seasonal_backgrounds': 'seasonal_backgrounds.csv',
    'seasonal_weather': 'seasonal_weather.csv',
    'seasonal_times': 'seasonal_times.csv',
    'additional_situations': 'additional_situations.csv',
    'general_composition': 'general_composition.csv',
    'gaze_direction': 'gaze_direction.csv',
    'poses': 'poses.csv',
    'body_directions': 'body_directions.csv',
}

def load_data_from_csv(file_path, skip_header=True):
    """Load data from CSV file into a dictionary."""
    data = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            if skip_header:
                next(csv_reader)  # Skip header row
            for row in csv_reader:
                if row and not row[0].startswith('#'):
                    key = row[0]
                    if key not in data:
                        data[key] = []
                    data[key].append(row[1:])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def load_list_from_csv(file_path):
    """Load list from CSV file, skipping header and comments."""
    return list(load_data_from_csv(file_path).keys())

def load_additional_situations(file_path):
    """Load additional situations from CSV file."""
    situations = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header row
            for row in csv_reader:
                if row and not row[0].startswith('#'):
                    situation = row[0]
                    conditions = row[1].split('|')
                    description = row[2]
                    situations[situation] = {"conditions": conditions, "description": description}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return situations

def get_random_element(lst):
    """Get a random element from a list."""
    return random.choice(lst)

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, DATA_DIR)
file_paths = {key: os.path.join(data_dir, value) for key, value in FILES.items()}

# Load data from CSV files
seasonal_fashion = load_data_from_csv(file_paths['seasonal_fashion'])
seasonal_backgrounds = load_data_from_csv(file_paths['seasonal_backgrounds'])
seasonal_weather = load_data_from_csv(file_paths['seasonal_weather'])
seasonal_times = load_data_from_csv(file_paths['seasonal_times'])
additional_situations = load_additional_situations(file_paths['additional_situations'])
general_composition = load_list_from_csv(file_paths['general_composition'])
gaze_direction = load_list_from_csv(file_paths['gaze_direction'])
poses = load_list_from_csv(file_paths['poses'])
body_directions = load_list_from_csv(file_paths['body_directions'])

class SeasonalFashionPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "season": (["random", "spring", "summer", "autumn", "winter"],),
                "background": (["random", "on", "off"],),
                "clothes": (["Yes", "No"], {"default": "Yes"}),
                "accessory": (["Yes", "No"], {"default": "Yes"}),
                "wanna_shot": (["Yes", "No"], {"default": "No", "label": "wanna shot?"}),
                "seed": ("INT", {"default": random.randint(0, 9999999999999999)}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"

    def generate_prompt(self, season, background, clothes, accessory, wanna_shot, seed):
        # Ensure the seed is an integer and set the random seed
        seed = int(str(seed)[:16])
        random.seed(seed)

        # Randomly select season and background if set to "random"
        if season == "random":
            season = get_random_element(["spring", "summer", "autumn", "winter"])
        if background == "random":
            background = get_random_element(["on", "off"])

        fashion = ""
        if clothes == "Yes":
            fashion = self.get_fashion(season, accessory)

        if wanna_shot == "Yes":
            return (", ".join(filter(None, [season, fashion, "cafe, coffee, holding a tumbler, straw inside the tumbler"])),)

        prompt_parts = [season, fashion, get_random_element(general_composition), 
                        get_random_element(gaze_direction), get_random_element(poses), 
                        get_random_element(body_directions)]

        if background == "on":
            background_prompt, additional_situation = self.get_background_and_situation(season)
            prompt_parts.extend([background_prompt, additional_situation])
        else:
            background_prompt = f"simple background, {get_random_element(['white background', 'grey background', 'black background'])}, standing cut"
            prompt_parts.append(background_prompt)

        # Join all prompt parts, filtering out any empty strings
        prompt = ", ".join(filter(None, prompt_parts))
        return (prompt,)

    def get_fashion(self, season, accessory):
        """Get a random fashion description for the given season."""
        fashion_category = seasonal_fashion.get(season, [])
        items = {
            "top": [item[1] for item in fashion_category if item[0] == "top"],
            "bottom": [item[1] for item in fashion_category if item[0] == "bottom"],
            "one_piece": [item[1] for item in fashion_category if item[0] == "one_piece"],
            "accessory": [item[1] for item in fashion_category if item[0] == "accessory"] if accessory == "Yes" else [],
            "hat": [item[1] for item in fashion_category if item[0] == "hat"],
            "shoes": [item[1] for item in fashion_category if item[0] == "shoes"],
            "socks": [item[1] for item in fashion_category if item[0] == "socks"],
        }

        top = get_random_element(items["top"]) if items["top"] else ""
        bottom = get_random_element(items["bottom"]) if items["bottom"] else ""
        one_piece = get_random_element(items["one_piece"]) if items["one_piece"] else ""
        accessory = get_random_element(items["accessory"]) if items["accessory"] else ""
        hat = get_random_element(items["hat"]) if items["hat"] else ""
        shoe = get_random_element(items["shoes"]) if items["shoes"] else ""
        sock = get_random_element(items["socks"]) if items["socks"] else ""

        if random.choice([True, False]) and (top or bottom):
            fashion = ", ".join(filter(None, [top, bottom, sock, accessory, hat, shoe]))
        else:
            fashion = ", ".join(filter(None, [one_piece, sock, accessory, hat, shoe]))

        return fashion

    def get_background_and_situation(self, season):
        """Get a background description and additional situation for the given season."""
        background_info = get_random_element(seasonal_backgrounds.get(season, []))
        background_classification = background_info[0]
        background = background_info[1]
        weather = get_random_element([item[0] for item in seasonal_weather.get(season, [])])
        time = get_random_element([item[0] for item in seasonal_times.get(season, [])])
        background_prompt = f"{background_classification}, {background}, {weather}, {time}"

        situation_details = [
            details["description"]
            for situation, details in additional_situations.items()
            if all(condition in [background_classification, weather, time] for condition in details["conditions"])
        ]

        additional_situation = ", ".join(situation_details)
        return background_prompt, additional_situation

# Example usage
node = SeasonalFashionPromptNode()
prompt = node.generate_prompt("random", "random", "Yes", "Yes", "No", random.randint(0, 9999999999999999))
print(prompt)
