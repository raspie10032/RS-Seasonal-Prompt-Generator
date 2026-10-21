import random
import csv
import os
import re

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

def load_data_from_csv(file_path, skip_header=True, min_columns=1):
    """Load data from CSV file into a dictionary."""
    data = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            if skip_header:
                next(csv_reader, None)  # Skip header row
            for row in csv_reader:
                if (len(row) >= min_columns and all(value.strip() for value in row[:min_columns])
                        and not row[0].lstrip().startswith('#')):
                    key = row[0].strip()
                    if key not in data:
                        data[key] = []
                    data[key].append([value.strip() for value in row[1:]])
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
            next(csv_reader, None)  # Skip header row
            for row in csv_reader:
                if row and row[0].strip() and not row[0].lstrip().startswith('#'):
                    if len(row) < 3:
                        continue
                    situation = row[0].strip()
                    conditions = [c.strip() for c in row[1].split('|') if c.strip()]
                    description = row[2]
                    situations[situation] = {"conditions": conditions, "description": description}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return situations

def get_random_element(lst, rng=None):
    """Get a random element from a list."""
    if not lst:
        return ""
    return (rng or random).choice(lst)

def situation_matches(conditions, context):
    """CSV `a&b|c` means (a AND b) OR c; bare words match whole words."""
    def has_term(term):
        return any(re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)",
                             value, re.IGNORECASE) for value in context if value)

    return any(all(has_term(term.strip()) for term in alternative.split('&'))
               for alternative in conditions
               if all(term.strip() for term in alternative.split('&')))

# Setting up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, DATA_DIR)
file_paths = {key: os.path.join(data_dir, value) for key, value in FILES.items()}

# Load data from CSV files
seasonal_fashion = load_data_from_csv(file_paths['seasonal_fashion'], min_columns=3)
seasonal_backgrounds = load_data_from_csv(file_paths['seasonal_backgrounds'], min_columns=3)
seasonal_weather = load_data_from_csv(file_paths['seasonal_weather'], min_columns=2)
seasonal_times = load_data_from_csv(file_paths['seasonal_times'], min_columns=2)
additional_situations = load_additional_situations(file_paths['additional_situations'])
general_composition = load_list_from_csv(file_paths['general_composition'])
gaze_direction = load_list_from_csv(file_paths['gaze_direction'])
poses = load_list_from_csv(file_paths['poses'])
body_directions = load_list_from_csv(file_paths['body_directions'])

class SeasonalFashionPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        # Use the same loaded tables for the UI and generation.
        def options(rows):
            return ["random", "off"] + list(dict.fromkeys(v for v in rows if v))

        weather_options = options(row[0] for rows in seasonal_weather.values()
                                  for row in rows if row)
        time_options = options(row[0] for rows in seasonal_times.values()
                               for row in rows if row)
        composition_options = options(general_composition)
        gaze_options = options(gaze_direction)
        pose_options = options(poses)
        body_direction_options = options(body_directions)

        return {
            "required": {
                "season": (["random", "spring", "summer", "autumn", "winter"],),
                "background_toggle": (["on", "off", "simple"],),
                # 날씨/시간 선택 옵션 확장
                "weather_toggle": (weather_options, {"default": "random"}),
                "time_toggle": (time_options, {"default": "random"}),
                # 패션 세부 요소 토글
                "top_toggle": (["random", "off"], {"default": "random"}),
                "bottom_toggle": (["random", "off"], {"default": "random"}),
                "one_piece_toggle": (["random", "off"], {"default": "random"}),
                "accessory_toggle": (["random", "off"], {"default": "random"}),
                "hat_toggle": (["random", "off"], {"default": "random"}),
                "shoes_toggle": (["random", "off"], {"default": "random"}),
                "socks_toggle": (["random", "off"], {"default": "random"}),
                # 포즈 및 구성 요소 토글 확장
                "composition_toggle": (composition_options, {"default": "random"}),
                "gaze_toggle": (gaze_options, {"default": "random"}),
                "pose_toggle": (pose_options, {"default": "random"}),
                "body_direction_toggle": (body_direction_options, {"default": "random"}),
                # 추가 상황 토글
                "additional_situation_toggle": (["on", "off"], {"default": "on"}),
                # 특수 케이스
                "wanna_shot": (["Yes", "No"], {"default": "No", "label": "wanna shot?"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "prompt"
    
    def generate_prompt(self, season, background_toggle, weather_toggle, time_toggle,
                         top_toggle, bottom_toggle, one_piece_toggle, accessory_toggle, hat_toggle,
                         shoes_toggle, socks_toggle, composition_toggle, gaze_toggle, pose_toggle,
                         body_direction_toggle, additional_situation_toggle, wanna_shot, seed):
        # A request-local RNG makes every integer seed reproducible and leaves
        # other ComfyUI nodes' random streams untouched.
        rng = random.Random(int(seed))

        # Randomly select season if set to "random"
        if season == "random":
            season = get_random_element(["spring", "summer", "autumn", "winter"], rng)

        # Handle special "wanna shot" case
        if wanna_shot == "Yes":
            fashion = self.get_fashion(season, top_toggle, bottom_toggle, one_piece_toggle,
                                       accessory_toggle, hat_toggle, shoes_toggle, socks_toggle, rng)
            shot_prompt = ", ".join(filter(None, [season, fashion, "cafe, coffee, holding a tumbler, straw inside the tumbler"]))
            return (shot_prompt,)

        # Build prompt parts based on toggle settings
        prompt_parts = [season]
        
        # Add fashion elements
        fashion = self.get_fashion(season, top_toggle, bottom_toggle, one_piece_toggle, 
                                  accessory_toggle, hat_toggle, shoes_toggle, socks_toggle, rng)
        if fashion:
            prompt_parts.append(fashion)
        
        # Add composition elements if enabled
        if composition_toggle != "off":
            if composition_toggle == "random":
                prompt_parts.append(get_random_element(general_composition, rng))
            else:
                prompt_parts.append(composition_toggle)
            
        if gaze_toggle != "off":
            if gaze_toggle == "random":
                prompt_parts.append(get_random_element(gaze_direction, rng))
            else:
                prompt_parts.append(gaze_toggle)
            
        if pose_toggle != "off":
            if pose_toggle == "random":
                prompt_parts.append(get_random_element(poses, rng))
            else:
                prompt_parts.append(pose_toggle)
            
        if body_direction_toggle != "off":
            if body_direction_toggle == "random":
                prompt_parts.append(get_random_element(body_directions, rng))
            else:
                prompt_parts.append(body_direction_toggle)

        # Add background elements if enabled
        if background_toggle == "on":
            background_prompt, additional_situation = self.get_background_and_situation(
                season, weather_toggle, time_toggle, additional_situation_toggle, rng)
            prompt_parts.append(background_prompt)
            if additional_situation:
                prompt_parts.append(additional_situation)
                
        elif background_toggle == "simple":
            background_prompt = f"simple background, {get_random_element(['white background', 'grey background', 'black background'], rng)}"
            prompt_parts.append(background_prompt)

        # Join all prompt parts, filtering out any empty strings
        prompt = ", ".join(filter(None, prompt_parts))
        return (prompt,)

    def get_fashion(self, season, top_toggle, bottom_toggle, one_piece_toggle, 
                   accessory_toggle, hat_toggle, shoes_toggle, socks_toggle, rng=None):
        """Get fashion description for the given season based on toggle settings."""
        fashion_category = seasonal_fashion.get(season, [])
        
        # Filter items based on toggles
        items = {
            "top": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "top"] if top_toggle != "off" else [],
            "bottom": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "bottom"] if bottom_toggle != "off" else [],
            "one_piece": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "one_piece"] if one_piece_toggle != "off" else [],
            "accessory": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "accessory"] if accessory_toggle != "off" else [],
            "hat": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "hat"] if hat_toggle != "off" else [],
            "shoes": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "shoes"] if shoes_toggle != "off" else [],
            "socks": [item[1] for item in fashion_category if len(item) >= 2 and item[0] == "socks"] if socks_toggle != "off" else [],
        }

        # Get random elements for each enabled category
        top = get_random_element(items["top"], rng)
        bottom = get_random_element(items["bottom"], rng)
        one_piece = get_random_element(items["one_piece"], rng)
        accessory = get_random_element(items["accessory"], rng)
        hat = get_random_element(items["hat"], rng)
        shoe = get_random_element(items["shoes"], rng)
        sock = get_random_element(items["socks"], rng)

        # Decide whether to use top+bottom or one_piece
        use_top_bottom = False
        if (top or bottom) and one_piece:
            use_top_bottom = (rng or random).choice([True, False])
        elif top or bottom:
            use_top_bottom = True
        elif one_piece:
            use_top_bottom = False
        
        # Build fashion string
        if use_top_bottom:
            fashion = ", ".join(filter(None, [top, bottom, sock, accessory, hat, shoe]))
        else:
            fashion = ", ".join(filter(None, [one_piece, sock, accessory, hat, shoe]))

        return fashion

    def get_background_and_situation(self, season, weather_toggle, time_toggle, additional_situation_toggle, rng=None):
        """Get background description and additional situation for the given season."""
        background_info = get_random_element([row for row in seasonal_backgrounds.get(season, [])
                                              if len(row) >= 2], rng)
        
        if not background_info:
            return "", ""
            
        background_classification = background_info[0]
        background = background_info[1]
        
        # Get weather based on toggle settings
        weather = ""
        if weather_toggle == "off":
            pass
        elif weather_toggle == "random":
            weather_items = [item[0] for item in seasonal_weather.get(season, [])]
            weather = get_random_element(weather_items, rng)
        else:
            # 특정 날씨가 선택된 경우
            weather = weather_toggle
            
        # Get time based on toggle settings
        time = ""
        if time_toggle == "off":
            pass
        elif time_toggle == "random":
            time_items = [item[0] for item in seasonal_times.get(season, [])]
            time = get_random_element(time_items, rng)
        else:
            # 특정 시간이 선택된 경우
            time = time_toggle
        
        # Build background prompt
        background_parts = [background_classification, background]
        if weather:
            background_parts.append(weather)
        if time:
            background_parts.append(time)
            
        background_prompt = ", ".join(filter(None, background_parts))
        
        # Add additional situation if enabled
        additional_situation = ""
        if additional_situation_toggle == "on":
            context = [season, background_classification, background, weather, time]
            situation_details = [
                details["description"]
                for situation, details in additional_situations.items()
                if situation_matches(details["conditions"], context)
            ]
            additional_situation = ", ".join(situation_details)
            
        return background_prompt, additional_situation

# Example usage
if __name__ == "__main__":
    node = SeasonalFashionPromptNode()
    prompt = node.generate_prompt(
        "random", "on", "random", "random",
        "random", "random", "random", "random", "random",
        "random", "random", "random", "random", "random",
        "random", "on", "No", random.randint(1000000000000000, 9999999999999999)
    )
    print(prompt)
