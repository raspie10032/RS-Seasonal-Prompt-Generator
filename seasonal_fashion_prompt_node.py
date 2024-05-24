import random
import csv
import os

def load_data_from_csv(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 행 건너뛰기 / Skip header row
        for row in csv_reader:
            if row and not row[0].startswith('#'):  # 주석 줄 무시 / Ignore comment lines
                key = row[0]
                if key not in data:
                    data[key] = []
                data[key].append(row[1:])
    return data

def load_list_from_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        return [row[0] for row in csv.reader(file) if row and not row[0].startswith('#')][1:]  # 주석 줄 무시, 헤더 행 건너뛰기 / Ignore comment lines, skip header row

def load_additional_situations(file_path):
    situations = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 행 건너뛰기 / Skip header row
        for row in csv_reader:
            if row and not row[0].startswith('#'):  # 주석 줄 무시 / Ignore comment lines
                situation = row[0]
                conditions = row[1].split('|')
                description = row[2]
                situations[situation] = {"conditions": conditions, "description": description}
    return situations

# 현재 디렉토리 확인
# Check the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")

# 절대 경로 설정
# Set absolute paths
data_dir = os.path.join(current_dir, 'data')
seasonal_fashion_path = os.path.join(data_dir, 'seasonal_fashion.csv')
seasonal_backgrounds_path = os.path.join(data_dir, 'seasonal_backgrounds.csv')
seasonal_weather_path = os.path.join(data_dir, 'seasonal_weather.csv')
seasonal_times_path = os.path.join(data_dir, 'seasonal_times.csv')
additional_situations_path = os.path.join(data_dir, 'additional_situations.csv')
general_composition_path = os.path.join(data_dir, 'general_composition.csv')
gaze_direction_path = os.path.join(data_dir, 'gaze_direction.csv')
poses_path = os.path.join(data_dir, 'poses.csv')
body_directions_path = os.path.join(data_dir, 'body_directions.csv')

# CSV 파일에서 데이터 로드
# Load data from CSV files
seasonal_fashion = load_data_from_csv(seasonal_fashion_path)
seasonal_backgrounds = load_data_from_csv(seasonal_backgrounds_path)
seasonal_weather = load_data_from_csv(seasonal_weather_path)
seasonal_times = load_data_from_csv(seasonal_times_path)
additional_situations = load_additional_situations(additional_situations_path)

general_composition = load_list_from_csv(general_composition_path)
gaze_direction = load_list_from_csv(gaze_direction_path)
poses = load_list_from_csv(poses_path)
body_directions = load_list_from_csv(body_directions_path)

# 리스트에서 랜덤 요소를 선택하는 함수
# Function to get a random element from a list
def get_random_element(lst):
    return random.choice(lst)

# 계절 패션 프롬프트 노드 클래스
# Class for Seasonal Fashion Prompt Node
class SeasonalFashionPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "season": (["random", "spring", "summer", "autumn", "winter"],),  # 계절 선택 / Season selection, including random option
                "seed": ("INT", {"default": random.randint(0, 9999999999999999)}),  # 최대 16자리 시드 / Up to 16-digit seed
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"

    # 프롬프트를 생성하는 함수
    # Function to generate a prompt
    def generate_prompt(self, season, seed):
        seed = int(str(seed)[:16])  # 시드를 최대 16자리로 제한 / Limit seed to maximum 16 digits
        random.seed(seed)  # 다른 결과를 보장하기 위해 시드 설정 / Set seed for different outcomes

        # 랜덤 계절 선택
        # Select a random season if 'random' is selected
        if season == "random":
            season = get_random_element(["spring", "summer", "autumn", "winter"])

        # 계절에 맞는 패션 아이템 로드
        # Load fashion items for the season
        fashion_category = seasonal_fashion[season]
        tops = [item[1] for item in fashion_category if item[0] == "top"]
        bottoms = [item[1] for item in fashion_category if item[0] == "bottom"]
        one_pieces = [item[1] for item in fashion_category if item[0] == "one_piece"]
        accessories = [item[1] for item in fashion_category if item[0] == "accessory"]
        hats = [item[1] for item in fashion_category if item[0] == "hat"]
        shoes = [item[1] for item in fashion_category if item[0] == "shoes"]
        socks = [item[1] for item in fashion_category if item[0] == "socks"]

        top = get_random_element(tops) if tops else ""
        bottom = get_random_element(bottoms) if bottoms else ""
        one_piece = get_random_element(one_pieces) if one_pieces else ""
        accessory = get_random_element(accessories) if accessories else ""
        hat = get_random_element(hats) if hats else ""
        shoe = get_random_element(shoes) if shoes else ""
        sock = get_random_element(socks) if socks else ""

        # 상의/하의 또는 원피스 선택
        # Select top/bottom or one_piece
        if random.choice([True, False]) and (top or bottom):
            fashion = ", ".join(filter(None, [top, bottom, sock, accessory, hat, shoe]))
        else:
            fashion = ", ".join(filter(None, [one_piece, sock, accessory, hat, shoe]))

        # 배경, 날씨, 시간 등의 정보 로드
        # Load background, weather, time, etc.
        background_info = get_random_element(seasonal_backgrounds[season])
        background_classification = background_info[0]
        background = background_info[1]
        weather = get_random_element([item[0] for item in seasonal_weather[season]])
        time = get_random_element([item[0] for item in seasonal_times[season]])
        composition = get_random_element(general_composition)
        gaze = get_random_element(gaze_direction)
        pose = get_random_element(poses)
        body_direction = get_random_element(body_directions)

        # 추가 상황 선택
        # Select additional situations
        situation_details = []
        for situation, details in additional_situations.items():
            conditions = details["conditions"]
            if all(condition in [background_classification, weather, time] for condition in conditions):
                situation_details.append(details["description"])

        additional_situation = ", ".join(situation_details)

        # 프롬프트 생성
        # Generate the prompt
        prompt = f"{season}, {fashion}, {composition}, {gaze}, {pose}, {body_direction}, {background_classification}, {background}, {weather}, {time}, {additional_situation}"

        return (prompt,)

# 예시 사용법
# Example usage
node = SeasonalFashionPromptNode()
prompt = node.generate_prompt("random", random.randint(0, 9999999999999999))  # 최대 16자리 시드 예시 / Example of up to 16-digit seed value
print(prompt)
