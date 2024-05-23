# This code was created by ChatGPT.
# 이 코드는 ChatGPT를 사용하여 만들어졌습니다.

import random
import csv
import os

def load_data_from_csv(file_path):
    data = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 헤더 행 건너뛰기
        for row in csv_reader:
            if row and not row[0].startswith('#'):  # 주석 줄 무시
                key = row[0]
                if key not in data:
                    data[key] = []
                data[key].append(row[1:])
    return data

def load_list_from_csv(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        return [row[0] for row in csv.reader(file) if row and not row[0].startswith('#')][1:]  # 주석 줄 무시, 헤더 행 건너뛰기

def load_additional_situations(file_path):
    situations = {}
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if row and not row[0].startswith('#'):  # 주석 줄 무시
                situation = row[0]
                conditions = row[1].split('|')
                situations[situation] = conditions
    return situations

# CSV 파일에서 데이터 로드
current_dir = os.path.dirname(os.path.abspath(__file__))
seasonal_fashion = load_data_from_csv(os.path.join(current_dir, 'data', 'seasonal_fashion.csv'))
seasonal_backgrounds = load_data_from_csv(os.path.join(current_dir, 'data', 'seasonal_backgrounds.csv'))
seasonal_weather = load_data_from_csv(os.path.join(current_dir, 'data', 'seasonal_weather.csv'))
seasonal_times = load_data_from_csv(os.path.join(current_dir, 'data', 'seasonal_times.csv'))

general_composition = load_list_from_csv(os.path.join(current_dir, 'data', 'general_composition.csv'))
gaze_direction = load_list_from_csv(os.path.join(current_dir, 'data', 'gaze_direction.csv'))
poses = load_list_from_csv(os.path.join(current_dir, 'data', 'poses.csv'))
body_directions = load_list_from_csv(os.path.join(current_dir, 'data', 'body_directions.csv'))

additional_situations = load_additional_situations(os.path.join(current_dir, 'data', 'additional_situations.csv'))

def get_random_element(lst):
    return random.choice(lst)

class SeasonalFashionPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "season": (["spring", "summer", "autumn", "winter"],),
                "seed": ("INT", {"default": 123456789012}),  # 기본 12자리 시드
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"

    def generate_prompt(self, season, seed):
        random.seed(seed)  # 다른 결과를 보장하기 위해 시드 설정

        fashion_category = seasonal_fashion[season]
        tops = [item[1] for item in fashion_category if item[0] == "top"]
        bottoms = [item[1] for item in fashion_category if item[0] == "bottom"]
        one_pieces = [item[1] for item in fashion_category if item[0] == "one_piece"]
        accessories = [item[1] for item in fashion_category if item[0] == "accessory"]
        hats = [item[1] for item in fashion_category if item[0] == "hat"]
        shoes = [item[1] for item in fashion_category if item[0] == "shoes"]

        top = get_random_element(tops) if tops else ""
        bottom = get_random_element(bottoms) if bottoms else ""
        one_piece = get_random_element(one_pieces) if one_pieces else ""
        accessory = get_random_element(accessories) if accessories else ""
        hat = get_random_element(hats) if hats else ""
        shoe = get_random_element(shoes) if shoes else ""

        # 상의/하의 또는 원피스 선택
        if random.choice([True, False]) and (top or bottom):
            fashion = ", ".join(filter(None, [top, bottom, accessory, hat, shoe]))
        else:
            fashion = ", ".join(filter(None, [one_piece, accessory, hat, shoe]))

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
        selected_situations = [situation for situation, conditions in additional_situations.items() 
                               if any(condition in [background, background_classification, weather, time] for condition in conditions)]

        additional_situation = ", ".join(selected_situations)
        prompt = f"{season}, {fashion}, {composition}, {gaze}, {pose}, {body_direction}, {background_classification}, {background}, {weather}, {time}, {additional_situation}"
        return (prompt,)

# 예시 사용법
node = SeasonalFashionPromptNode()
prompt = node.generate_prompt("summer", 123456789012)
print(prompt)
