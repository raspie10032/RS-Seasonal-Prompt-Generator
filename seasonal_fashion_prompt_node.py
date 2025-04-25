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
    if not lst:
        return ""
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
        # 미리 데이터 로드하기
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, DATA_DIR)
        file_paths = {key: os.path.join(data_dir, value) for key, value in FILES.items()}
        
        # 데이터 로드
        weather_options = ["random", "off"]
        time_options = ["random", "off"]
        composition_options = ["random", "off"]
        gaze_options = ["random", "off"]
        pose_options = ["random", "off"]
        body_direction_options = ["random", "off"]
        
        try:
            # 날씨 옵션 로드
            weather_data = {}
            with open(file_paths['seasonal_weather'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        season = row[0]
                        weather = row[1]
                        if weather not in weather_options and weather != "":
                            weather_options.append(weather)
            
            # 시간 옵션 로드
            with open(file_paths['seasonal_times'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        season = row[0]
                        time = row[1]
                        if time not in time_options and time != "":
                            time_options.append(time)
            
            # 구성 옵션 로드
            with open(file_paths['general_composition'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        comp = row[0]
                        if comp not in composition_options and comp != "":
                            composition_options.append(comp)
            
            # 시선 방향 옵션 로드
            with open(file_paths['gaze_direction'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        gaze = row[0]
                        if gaze not in gaze_options and gaze != "":
                            gaze_options.append(gaze)
            
            # 포즈 옵션 로드
            with open(file_paths['poses'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        pose = row[0]
                        if pose not in pose_options and pose != "":
                            pose_options.append(pose)
            
            # 몸 방향 옵션 로드
            with open(file_paths['body_directions'], mode='r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip header
                for row in csv_reader:
                    if row and not row[0].startswith('#'):
                        direction = row[0]
                        if direction not in body_direction_options and direction != "":
                            body_direction_options.append(direction)
                            
        except Exception as e:
            print(f"Error loading options: {e}")
        
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
                "seed": ("INT", {"default": random.randint(1000000000000000, 9999999999999999)}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = "prompt"
    
    def generate_prompt(self, season, background_toggle, weather_toggle, time_toggle, 
                         top_toggle, bottom_toggle, one_piece_toggle, accessory_toggle, hat_toggle, 
                         shoes_toggle, socks_toggle, composition_toggle, gaze_toggle, pose_toggle, 
                         body_direction_toggle, additional_situation_toggle, wanna_shot, seed):
        # Convert seed to integer
        seed = int(seed)
        
        # Define the range for 16-digit seeds
        MIN_SEED = 1000000000000000  # 10^15
        MAX_SEED = 9999999999999999  # 10^16 - 1
        
        # Process seed value based on range
        if seed < MIN_SEED:
            # For values below minimum, multiply by a random 16-digit number and extract 16 digits
            multiplier = random.randint(MIN_SEED, MAX_SEED)
            product = seed * multiplier
            # Extract 16 digits
            seed_str = str(product)
            if len(seed_str) >= 16:
                # Take the last 16 digits if result has more than 16 digits
                seed = int(seed_str[-16:])
            else:
                # If result is less than 16 digits, square the number and extract again
                squared = product * product
                seed_str = str(squared)
                if len(seed_str) >= 16:
                    seed = int(seed_str[-16:])
                else:
                    # If still less than 16 digits, use the original multiplier as a fallback
                    seed = multiplier
        elif seed > MAX_SEED:
            # For values above maximum, extract 16 digits
            seed_str = str(seed)
            # Take the last 16 digits
            seed = int(seed_str[-16:])
            
        # Ensure seed is within range after processing
        seed = max(MIN_SEED, min(seed, MAX_SEED))
        random.seed(seed)

        # Randomly select season if set to "random"
        if season == "random":
            season = get_random_element(["spring", "summer", "autumn", "winter"])

        # Handle special "wanna shot" case
        if wanna_shot == "Yes":
            fashion = self.get_fashion(season, top_toggle, bottom_toggle, one_piece_toggle, 
                                       accessory_toggle, hat_toggle, shoes_toggle, socks_toggle)
            return (", ".join(filter(None, [season, fashion, "cafe, coffee, holding a tumbler, straw inside the tumbler"])),)

        # Build prompt parts based on toggle settings
        prompt_parts = [season]
        
        # Add fashion elements
        fashion = self.get_fashion(season, top_toggle, bottom_toggle, one_piece_toggle, 
                                  accessory_toggle, hat_toggle, shoes_toggle, socks_toggle)
        if fashion:
            prompt_parts.append(fashion)
        
        # Add composition elements if enabled
        if composition_toggle != "off":
            if composition_toggle == "random":
                prompt_parts.append(get_random_element(general_composition))
            else:
                prompt_parts.append(composition_toggle)
            
        if gaze_toggle != "off":
            if gaze_toggle == "random":
                prompt_parts.append(get_random_element(gaze_direction))
            else:
                prompt_parts.append(gaze_toggle)
            
        if pose_toggle != "off":
            if pose_toggle == "random":
                prompt_parts.append(get_random_element(poses))
            else:
                prompt_parts.append(pose_toggle)
            
        if body_direction_toggle != "off":
            if body_direction_toggle == "random":
                prompt_parts.append(get_random_element(body_directions))
            else:
                prompt_parts.append(body_direction_toggle)

        # Add background elements if enabled
        if background_toggle == "on":
            background_prompt, additional_situation = self.get_background_and_situation(
                season, weather_toggle, time_toggle, additional_situation_toggle)
            prompt_parts.append(background_prompt)
            if additional_situation:
                prompt_parts.append(additional_situation)
                
        elif background_toggle == "simple":
            background_prompt = f"simple background, {get_random_element(['white background', 'grey background', 'black background'])}, standing cut"
            prompt_parts.append(background_prompt)

        # Join all prompt parts, filtering out any empty strings
        prompt = ", ".join(filter(None, prompt_parts))
        return (prompt,)

    def get_fashion(self, season, top_toggle, bottom_toggle, one_piece_toggle, 
                   accessory_toggle, hat_toggle, shoes_toggle, socks_toggle):
        """Get fashion description for the given season based on toggle settings."""
        fashion_category = seasonal_fashion.get(season, [])
        
        # Filter items based on toggles
        items = {
            "top": [item[1] for item in fashion_category if item[0] == "top"] if top_toggle != "off" else [],
            "bottom": [item[1] for item in fashion_category if item[0] == "bottom"] if bottom_toggle != "off" else [],
            "one_piece": [item[1] for item in fashion_category if item[0] == "one_piece"] if one_piece_toggle != "off" else [],
            "accessory": [item[1] for item in fashion_category if item[0] == "accessory"] if accessory_toggle != "off" else [],
            "hat": [item[1] for item in fashion_category if item[0] == "hat"] if hat_toggle != "off" else [],
            "shoes": [item[1] for item in fashion_category if item[0] == "shoes"] if shoes_toggle != "off" else [],
            "socks": [item[1] for item in fashion_category if item[0] == "socks"] if socks_toggle != "off" else [],
        }

        # Get random elements for each enabled category
        top = get_random_element(items["top"])
        bottom = get_random_element(items["bottom"])
        one_piece = get_random_element(items["one_piece"])
        accessory = get_random_element(items["accessory"])
        hat = get_random_element(items["hat"])
        shoe = get_random_element(items["shoes"])
        sock = get_random_element(items["socks"])

        # Decide whether to use top+bottom or one_piece
        use_top_bottom = False
        if top and bottom and one_piece:
            use_top_bottom = random.choice([True, False])
        elif top and bottom:
            use_top_bottom = True
        elif one_piece:
            use_top_bottom = False
        
        # Build fashion string
        if use_top_bottom:
            fashion = ", ".join(filter(None, [top, bottom, sock, accessory, hat, shoe]))
        else:
            fashion = ", ".join(filter(None, [one_piece, sock, accessory, hat, shoe]))

        return fashion

    def get_background_and_situation(self, season, weather_toggle, time_toggle, additional_situation_toggle):
        """Get background description and additional situation for the given season."""
        background_info = get_random_element(seasonal_backgrounds.get(season, []))
        
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
            weather = get_random_element(weather_items)
        else:
            # 특정 날씨가 선택된 경우
            weather = weather_toggle
            
        # Get time based on toggle settings
        time = ""
        if time_toggle == "off":
            pass
        elif time_toggle == "random":
            time_items = [item[0] for item in seasonal_times.get(season, [])]
            time = get_random_element(time_items)
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
        if additional_situation_toggle == "on" and background_classification and (weather or time):
            conditions = [c for c in [background_classification, weather, time] if c]
            situation_details = [
                details["description"]
                for situation, details in additional_situations.items()
                if any(condition in details["conditions"] for condition in conditions)
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