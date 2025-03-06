# app.py

import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, url_for, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import json
import plotly.graph_objects as go
from flask_cors import CORS
import plotly
app = Flask(__name__)
CORS(app)

# Direktori untuk menyimpan model dan visualisasi
MODEL_DIR = 'models'
TREE_VIS_DIR = 'tree_visualization'
DATA_DIR = 'data'
STATIC_TREE_VIS_DIR = os.path.join('static', 'tree_visualization')
VISUALIZATION_DIR = os.path.join('static', 'tree_visualization')

# Membuat direktori jika belum ada
for directory in [MODEL_DIR, TREE_VIS_DIR, DATA_DIR, STATIC_TREE_VIS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Mapping label untuk visualisasi
label_mapping = {
    0: "Sangat Buruk",
    1: "Buruk",
    2: "Cukup",
    3: "Baik",
    4: "Sangat Baik"
}

# Mapping fitur ke label kategorinya
label_mapping_per_feature = {
    'PAI': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'PKn': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'BIN': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'MTK': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'SEI': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'BIG': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'SEB': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'PJOK': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'PRA': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'GEO': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'SEJ': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'Sos': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'Eko': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'AA': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'TIK': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"],
    'BDS': ["Sangat Buruk", "Buruk", "Cukup", "Baik", "Sangat Baik"]
}

# Fungsi untuk mengganti label numerik dengan label asli
def replace_labels(feature_name, value):
    if feature_name in label_mapping_per_feature:
        return label_mapping_per_feature[feature_name][value]
    return value

# Fungsi untuk mengekstrak aturan dari pohon keputusan
def extract_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # Split kondisi
            path_left = path.copy()
            path_left.append(f"{name} <= {threshold:.2f}")
            recurse(tree_.children_left[node], path_left, paths)

            path_right = path.copy()
            path_right.append(f"{name} > {threshold:.2f}")
            recurse(tree_.children_right[node], path_right, paths)
        else:
            # Leaf node
            value = tree_.value[node]
            class_idx = value.argmax()
            class_name = class_names[class_idx]
            paths.append((path, class_name))

    recurse(0, path, paths)
    return paths

# Fungsi untuk mengonversi kondisi numerik ke kategori
def convert_conditions_to_categories(conditions):
    new_conditions = []
    for cond in conditions:
        if " <= " in cond:
            feature, _, threshold = cond.partition(" <= ")
            # Dengan One-Hot Encoding, fitur bernama 'MTK_Baik', 'MTK_Cukup', dll.
            # Kondisi 'MTK_Baik <= 0.5' berarti fitur 'MTK_Baik' adalah 0, sehingga 'MTK' tidak 'Baik'
            # Sebaliknya, jika 'MTK_Baik > 0.5', berarti 'MTK' adalah 'Baik'
            if "_Sangat Buruk" in feature:
                category = "Sangat Buruk"
                new_conditions.append(f"{feature.split('_')[0]} != \"{category}\"")
            elif "_Buruk" in feature:
                category = "Buruk"
                new_conditions.append(f"{feature.split('_')[0]} != \"{category}\"")
            elif "_Cukup" in feature:
                category = "Cukup"
                new_conditions.append(f"{feature.split('_')[0]} != \"{category}\"")
            elif "_Baik" in feature:
                category = "Baik"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            elif "_Sangat Baik" in feature:
                category = "Sangat Baik"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            else:
                # Jika fitur tidak mengikuti pola One-Hot Encoding, gunakan kondisi numerik
                new_conditions.append(cond)
        elif " > " in cond:
            feature, _, threshold = cond.partition(" > ")
            if "_Sangat Buruk" in feature:
                category = "Sangat Buruk"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            elif "_Buruk" in feature:
                category = "Buruk"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            elif "_Cukup" in feature:
                category = "Cukup"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            elif "_Baik" in feature:
                category = "Baik"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            elif "_Sangat Baik" in feature:
                category = "Sangat Baik"
                new_conditions.append(f"{feature.split('_')[0]} = \"{category}\"")
            else:
                new_conditions.append(cond)
        else:
            new_conditions.append(cond)
    return new_conditions

# Fungsi untuk mengonversi aturan menjadi format IF ... THEN ...
def format_rule(conditions, class_name):
    if not conditions:
        return f"THEN CLASS \"{class_name}\""
    else:
        return f"IF {' AND '.join(conditions)} THEN CLASS \"{class_name}\""

@app.route('/')
def index():
    api_endpoints = {
        "endpoints": {
            "Upload Data Latih": {"method": "POST", "url": "/upload"},
            "Get Data Latih": {"method": "GET", "url": "/data_latih"},
            "Train Model": {"method": "GET", "url": "/training"},
            "Evaluate Model": {"method": "GET", "url": "/evaluating"},
            "Get Rules": {"method": "GET", "url": "/rules"},
            "Get Visualizations": {"method": "GET", "url": "/visualizations"},
            "Get Tree Image": {"method": "GET", "url": "/tree_visualization/<filename>"},
            "Predict": {"method": "POST", "url": "/predict"}
        }
    }
    return jsonify(api_endpoints), 200
def manual_evaluation_metrics(y_true, y_pred):
    """
    Menghitung metrik evaluasi (accuracy, precision, recall, f1-score) secara manual.
    y_true: list/array dari nilai aktual
    y_pred: list/array dari nilai prediksi
    """
    # Mendapatkan kelas unik
    classes = sorted(set(y_true) | set(y_pred))
    n_samples = len(y_true)
    
    # Inisialisasi confusion matrix sebagai dictionary
    confusion_matrix = {c: {pred_c: 0 for pred_c in classes} for c in classes}
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true][pred] += 1
    
    # Menghitung accuracy
    correct_predictions = sum(confusion_matrix[c][c] for c in classes)
    accuracy = correct_predictions / n_samples if n_samples > 0 else 0
    
    # Menghitung precision, recall, dan F1-score per kelas, lalu rata-rata
    precision_dict = {}
    recall_dict = {}
    f1_dict = {}
    support_dict = {}
    
    for c in classes:
        # True Positives (TP), False Positives (FP), False Negatives (FN)
        TP = confusion_matrix[c][c]
        FP = sum(confusion_matrix[other_c][c] for other_c in classes if other_c != c)
        FN = sum(confusion_matrix[c][other_c] for other_c in classes if other_c != c)
        TN = sum(confusion_matrix[other_c][other_pred] 
                 for other_c in classes if other_c != c 
                 for other_pred in classes if other_pred != c)
        
        # Precision = TP / (TP + FP)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        precision_dict[c] = precision
        
        # Recall = TP / (TP + FN)
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        recall_dict[c] = recall
        
        # F1-score = 2 * (precision * recall) / (precision + recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_dict[c] = f1
        
        # Support (jumlah kemunculan kelas aktual)
        support_dict[c] = TP + FN
    
    # Menghitung rata-rata weighted untuk precision, recall, dan F1-score
    total_support = sum(support_dict.values())
    precision_weighted = sum(precision_dict[c] * support_dict[c] for c in classes) / total_support if total_support > 0 else 0
    recall_weighted = sum(recall_dict[c] * support_dict[c] for c in classes) / total_support if total_support > 0 else 0
    f1_weighted = sum(f1_dict[c] * support_dict[c] for c in classes) / total_support if total_support > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1_score": f1_weighted
    }

# Routing tunggal dengan metode POST
def extract_rules(rf_model, feature_names, class_names):
    rules = []
    for idx, estimator in enumerate(rf_model.estimators_[:5]):  # Batasi ke 5 pohon untuk efisiensi
        tree = estimator.tree_
        feature = tree.feature
        threshold = tree.threshold
        children_left = tree.children_left
        children_right = tree.children_right
        value = tree.value
        
        def recurse(node, conditions):
            if children_left[node] == -1 and children_right[node] == -1:  # Leaf node
                class_idx = value[node].argmax()
                class_name = class_names[class_idx]
                rule = f"JIKA {' DAN '.join(conditions)} MAKA Status Prestasi = {class_name}"
                rules.append(rule)
            else:
                feat_idx = feature[node]
                if feat_idx != -2:  # Bukan leaf
                    feat_name = feature_names[feat_idx]
                    thresh = threshold[node]
                    
                    # Kondisi untuk cabang kiri (<= threshold) berarti FALSE
                    left_conditions = conditions + [f"{feat_name} = FALSE"]
                    recurse(children_left[node], left_conditions)
                    
                    # Kondisi untuk cabang kanan (> threshold) berarti TRUE
                    right_conditions = conditions + [f"{feat_name} = TRUE"]
                    recurse(children_right[node], right_conditions)
        
        recurse(0, [])
    return rules

@app.route('/get_all_data', methods=['GET'])
def get_all_data():
    try:
        # Membaca data dari file
        file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
        if not os.path.exists(file_path):
            return jsonify({"error": "Data belum diunggah."}), 400
        data = pd.read_excel(file_path)
        
        # Simpan data asli dan konversi ke format JSON
        all_data_raw = data.to_dict(orient='records')
        
        # Mengembalikan hasil dalam bentuk JSON
        response = {
            "message": "Berhasil mengambil seluruh data.",
            "all_data": all_data_raw
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": f"Error saat memproses: {str(e)}"}), 500
# Routing tunggal dengan metode POST
@app.route('/process', methods=['POST'])
def process():
    try:
        # Menerima persentase data latih dari request
        data = request.get_json()
        train_percentage = data.get('train_percentage', None)
        
        # Validasi input
        if train_percentage is None:
            return jsonify({"error": "Parameter 'train_percentage' harus disertakan."}), 400
        if not isinstance(train_percentage, (int, float)) or train_percentage < 0 or train_percentage > 100:
            return jsonify({"error": "Persentase data latih harus berupa angka antara 0 dan 100."}), 400
        
        # Membaca data dari file
        file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
        if not os.path.exists(file_path):
            return jsonify({"error": "Data belum diunggah."}), 400
        data = pd.read_excel(file_path)
        
        # Simpan data asli untuk informasi NIS dan NAMA SISWA
        data_raw = data.copy()
        
        # Preprocessing Data
        data_cleaned = data.drop(columns=['NO', 'NAMA SISWA', 'NISN', 'NIS'], errors='ignore')
        
        # Encoding variabel target 'Status Prestasi' dengan LabelEncoder
        label_encoder = LabelEncoder()
        data_cleaned['Status Prestasi'] = label_encoder.fit_transform(data_cleaned['Status Prestasi'])
        
        # One-Hot Encoding untuk fitur kategorikal
        categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
        data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features)
        
        # Pisahkan fitur dan target
        X = data_encoded.drop(columns=['Status Prestasi'])
        y = data_encoded['Status Prestasi']
        
        # Membagi data berdasarkan persentase
        test_size = 1 - (train_percentage / 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Mengambil indeks untuk data latih dan data uji
        train_indices = X_train.index
        test_indices = X_test.index
        
        # Data latih dan data uji dalam bentuk asli
        train_data_raw = data_raw.loc[train_indices].to_dict(orient='records')
        test_data_raw = data_raw.loc[test_indices]
        
        # Melatih model Random Forest
        rf_model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Simpan model
        model_filename = os.path.join(MODEL_DIR, "random_forest_model.pkl")
        joblib.dump(rf_model, model_filename)
        
        # Ekstrak aturan dengan format TRUE/FALSE
        feature_names = X.columns.tolist()
        class_names = label_encoder.classes_.tolist()
        rules = extract_rules(rf_model, feature_names, class_names)
        rules_json_filename = os.path.join(MODEL_DIR, 'rules.json')
        with open(rules_json_filename, 'w') as f:
            json.dump(rules, f, indent=4)
        
        # Membuat visualisasi pohon keputusan dari salah satu estimator
        plt.figure(figsize=(20, 10))
        plot_tree(
            rf_model.estimators_[0],
            feature_names=feature_names,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        visualization_filename = 'decision_tree.png'
        visualization_path = os.path.join(VISUALIZATION_DIR, visualization_filename)
        plt.savefig(visualization_path)
        plt.close()
        
        # URL untuk mengakses visualisasi
        visualization_url = url_for('static', filename=f'tree_visualization/{visualization_filename}', _external=True)
        
        # Melakukan prediksi pada data latih dan data uji
        y_train_pred = rf_model.predict(X_train)
        y_test_pred = rf_model.predict(X_test)
        
        # Mengembalikan hasil prediksi ke label asli
        y_test_pred_labels = label_encoder.inverse_transform(y_test_pred)
        
        # Membuat daftar hasil prediksi untuk data uji dengan NIS, NAMA SISWA, dan Hasil Prediksi
        predictions = []
        for idx, pred_label in zip(test_indices, y_test_pred_labels):
            row = test_data_raw.loc[idx]
            predictions.append({
                "NIS": row.get('NIS', 'N/A'),
                "NAMA SISWA": row.get('NAMA SISWA', 'N/A'),
                "Hasil Prediksi": pred_label
            })
        
        # Evaluasi manual untuk data latih
        train_evaluation = manual_evaluation_metrics(y_train, y_train_pred)
        
        # Evaluasi manual untuk data uji
        test_evaluation = manual_evaluation_metrics(y_test, y_test_pred)
        
        # Mengembalikan hasil dalam bentuk JSON
        response = {
            "message": "Model berhasil dilatih, aturan diekstrak, visualisasi dibuat, dan dievaluasi.",
            "train_percentage": train_percentage,
            "test_percentage": 100 - train_percentage,
            "training_data": train_data_raw,  # Semua data latih
            "predictions": predictions,  # NIS, NAMA SISWA, dan Hasil Prediksi untuk data uji
            "evaluation": {
                "train": train_evaluation,
                "test": test_evaluation
            },
            "rules": rules,  # Aturan dalam format TRUE/FALSE
            "visualization": visualization_url  # URL ke gambar pohon keputusan
        }
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({"error": f"Error saat memproses: {str(e)}"}), 500
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    if file and file.filename.endswith('.xlsx'):
        file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
        file.save(file_path)
        return jsonify({"message": "File successfully uploaded"}), 200
    else:
        return jsonify({"error": "Allowed file type is .xlsx"}), 400

@app.route('/data_latih', methods=['GET'])
def data_latih():
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    if not os.path.exists(file_path):
        return jsonify({"error": "Data latih belum diunggah."}), 400
    try:
        data = pd.read_excel(file_path)
        data_preview = data.head().to_dict(orient='records')
        return jsonify({"data_preview": data_preview}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read data latih: {str(e)}"}), 500

@app.route('/training', methods=['GET'])
def training():
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    if not os.path.exists(file_path):
        return jsonify({"error": "Data latih belum diunggah."}), 400
    try:
        data = pd.read_excel(file_path)

        # Menggunakan hanya 80% data dari main_data.xlsx
        data = data.sample(frac=0.8, random_state=42).reset_index(drop=True)

        # Preprocessing Data
        data_cleaned = data.drop(columns=['NO', 'NAMA SISWA', 'NISN', 'NIS'])

        # Encoding variabel kategori 'Status Prestasi' dengan LabelEncoder
        label_encoder = LabelEncoder()
        data_cleaned['Status Prestasi'] = label_encoder.fit_transform(data_cleaned['Status Prestasi'])

        # One-Hot Encoding untuk fitur kategorikal
        categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
        data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features)

        # Menyimpan nama-nama fitur yang telah di-encode untuk visualisasi nanti
        encoded_feature_names = data_encoded.drop(columns=['Status Prestasi']).columns.tolist()

        # Pisahkan fitur dan target
        X = data_encoded.drop(columns=['Status Prestasi'])
        y = data_encoded['Status Prestasi']

        # Membagi Data ke Training dan Testing (tetap 70-30 dari 80% data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Membentuk Model Random Forest dengan Max Depth Maksimal
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)

        # Menyimpan model ke file
        model_filename = os.path.join(MODEL_DIR, "random_forest_model_max_depth.pkl")
        joblib.dump(rf_model, model_filename)

        # Menyimpan nama-nama fitur yang telah di-encode
        encoder_filename = os.path.join(MODEL_DIR, "encoded_feature_names.pkl")
        joblib.dump(encoded_feature_names, encoder_filename)

        # Menyimpan label encoder untuk target
        target_encoder_filename = os.path.join(MODEL_DIR, "target_encoder.pkl")
        joblib.dump(label_encoder, target_encoder_filename)

        # Ekstrak aturan dari setiap pohon dalam Random Forest
        all_rules = []
        readable_rules = []

        # Batasi jumlah pohon yang divisualisasikan
        MAX_TREES_TO_VISUALIZE = 300
        trees_to_visualize = rf_model.estimators_[:MAX_TREES_TO_VISUALIZE]

        for i, tree in enumerate(trees_to_visualize):
            rules = extract_rules(tree, encoded_feature_names, label_encoder.classes_)
            for conditions, class_name in rules:
                # Konversi kondisi numerik ke kategori
                readable_conditions = convert_conditions_to_categories(conditions)
                # Format aturan
                rule_text = format_rule(readable_conditions, class_name)
                readable_rules.append(rule_text)
                # Simpan aturan dalam format yang dapat digunakan untuk prediksi
                all_rules.append({
                    "conditions": readable_conditions,
                    "class": class_name
                })

            # Visualisasi pohon
            plt.figure(figsize=(30, 15))
            plot_tree(
                tree,
                filled=True,
                feature_names=encoded_feature_names,            # Pastikan ini adalah list
                class_names=label_encoder.classes_.tolist(),    # Pastikan ini adalah list
                rounded=True,
                proportion=False,
                precision=2,
                fontsize=10
            )
            tree_image_filename = f"tree_{i+1}.png"
            tree_image_path = os.path.join(STATIC_TREE_VIS_DIR, tree_image_filename)
            plt.savefig(tree_image_path)
            plt.close()

        # Simpan aturan ke dalam file teks dengan pemisahan per aturan
        rules_text_filename = os.path.join(TREE_VIS_DIR, 'rules_readable.txt')
        with open(rules_text_filename, 'w') as f:
            for idx, rule in enumerate(readable_rules, 1):
                f.write(f"Rule {idx}:\n{rule}\n\n")

        # Simpan aturan ke dalam file JSON
        rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')
        with open(rules_json_filename, 'w') as f:
            json.dump(all_rules, f, indent=4)

        return jsonify({
            "message": "Model berhasil dilatih dengan 80% data dan disimpan.",
            "rules_text": "rules_readable.txt",
            "rules_json": "rules.json",
            "visualized_trees": len(trees_to_visualize)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error saat melatih model: {str(e)}"}), 500



@app.route('/rules', methods=['GET'])
def rules():
    rules_text_filename = os.path.join(TREE_VIS_DIR, 'rules_readable.txt')
    rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')
    if not os.path.exists(rules_text_filename) or not os.path.exists(rules_json_filename):
        return jsonify({"error": "Rules belum tersedia. Silakan latih model terlebih dahulu."}), 400
    try:
        with open(rules_text_filename, 'r') as f:
            rules_content = f.read()
        with open(rules_json_filename, 'r') as f:
            rules_json = json.load(f)
        return jsonify({"rules_text": rules_content, "rules_json": rules_json}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read rules: {str(e)}"}), 500

@app.route('/visualizations', methods=['GET'])
def visualizations():
    try:
        images = [img for img in os.listdir(STATIC_TREE_VIS_DIR) if img.endswith('.png')]
        image_urls = [url_for('tree_visualization', filename=img, _external=True) for img in images]
        if not images:
            return jsonify({"message": "Visualisasi pohon belum tersedia. Silakan latih model terlebih dahulu."}), 200
        return jsonify({"visualizations": image_urls}), 200
    except Exception as e:
        return jsonify({"error": f"Error saat memuat visualisasi: {str(e)}"}), 500

@app.route('/tree_visualization/<filename>', methods=['GET'])
def tree_visualization(filename):
    try:
        return send_from_directory(STATIC_TREE_VIS_DIR, filename)
    except Exception as e:
        return jsonify({"error": f"File tidak ditemukan: {str(e)}"}), 404

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({"error": "Invalid input data"}), 400

        # Normalisasi input_data: ubah semua key dan value menjadi huruf kecil dan hapus spasi
        normalized_input = {k.strip().lower(): v.strip().lower() for k, v in input_data.items()}

        # Memuat aturan dari JSON
        rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')
        if not os.path.exists(rules_json_filename):
            return jsonify({"error": "Rules tidak ditemukan. Silakan latih model terlebih dahulu."}), 400
        with open(rules_json_filename, 'r') as f:
            rules = json.load(f)

        votes = {}
        matched_rules = 0  # Untuk melacak jumlah aturan yang cocok

        for rule in rules:
            conditions = rule['conditions']
            class_name = rule['class']
            match = True
            for condition in conditions:
                # Condition format: "Fitur = "Kategori"" atau "Fitur != "Kategori""
                if " = " in condition:
                    feature, _, value = condition.partition(" = ")
                    feature = feature.strip().lower()
                    value = value.strip('"').lower()
                    if feature not in normalized_input or normalized_input[feature] != value:
                        match = False
                        break
                elif " != " in condition:
                    feature, _, value = condition.partition(" != ")
                    feature = feature.strip().lower()
                    value = value.strip('"').lower()
                    if feature not in normalized_input or normalized_input[feature] == value:
                        match = False
                        break
                else:
                    # Format kondisi tidak didukung
                    match = False
                    break
            if match:
                votes[class_name] = votes.get(class_name, 0) + 1
                matched_rules += 1

        if matched_rules == 0:
            return jsonify({"error": "No rules matched the input data."}), 400

        # Tentukan kelas dengan suara terbanyak
        predicted_class = max(votes, key=votes.get)

        return jsonify({
            "predicted_class": predicted_class,
            "votes": votes,
            "matched_rules": matched_rules
        }), 200

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500


@app.route('/data_uji', methods=['GET'])
def data_uji():
    """
    Mengembalikan data uji dalam format JSON.
    """
    file_path = os.path.join(DATA_DIR, 'data_uji.xlsx')
    if not os.path.exists(file_path):
        return jsonify({"error": "Data uji belum diunggah."}), 400
    try:
        data = pd.read_excel(file_path)
        data_preview = data.to_dict(orient='records')
        return jsonify({"data_uji": data_preview}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to read data uji: {str(e)}"}), 500


@app.route('/web/predict_data_uji', methods=['GET'])
def web_predict_data_uji():
    """
    Menampilkan halaman web yang berisi hasil prediksi data uji
    serta jumlah siswa berprestasi, cukup berprestasi, dan tidak berprestasi
    dalam bentuk card dan grafik, hanya menggunakan 20% data dari baris paling akhir.
    """
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')  # Menggunakan main_data.xlsx
    rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')

    if not os.path.exists(file_path):
        return render_template('predict_data_uji.html', error="Data uji belum diunggah.")

    if not os.path.exists(rules_json_filename):
        return render_template('predict_data_uji.html', error="Rules tidak ditemukan. Silakan latih model terlebih dahulu.")

    try:
        # Membaca data dari main_data.xlsx
        data = pd.read_excel(file_path)
        if data.empty:
            return render_template('predict_data_uji.html', error="Data uji kosong.")

        # Ambil hanya 20% baris terakhir
        num_rows = int(len(data) * 0.2)  # Hitung jumlah baris untuk 20%
        data = data.tail(num_rows).reset_index(drop=True)  # Ambil 20% terakhir

        # Simpan data mentah (sudah dalam format kategorikal)
        data_raw = data.copy()

        # Data sudah dalam bentuk kategorikal, jadi langsung gunakan
        data_labeled = data.copy()

        # Memuat aturan
        with open(rules_json_filename, 'r') as f:
            rules = json.load(f)

        predictions = []
        counts = {"Berprestasi": 0, "Cukup Berprestasi": 0, "Tidak Berprestasi": 0}

        # Proses prediksi
        for index, row in data_labeled.iterrows():
            raw_data = data_raw.iloc[index].to_dict()  # Menyimpan nilai mentah
            normalized_input = {str(k).strip().lower(): str(v).strip().lower() for k, v in row.items()}
            votes = {}
            matched_rules = 0

            for rule in rules:
                conditions = rule['conditions']
                class_name = rule['class']
                match = True

                for condition in conditions:
                    if " = " in condition:
                        feature, _, value = condition.partition(" = ")
                        feature = feature.strip().lower()
                        value = value.strip('"').lower()
                        if feature not in normalized_input or normalized_input[feature] != value:
                            match = False
                            break
                    elif " != " in condition:
                        feature, _, value = condition.partition(" != ")
                        feature = feature.strip().lower()
                        value = value.strip('"').lower()
                        if feature not in normalized_input or normalized_input[feature] == value:
                            match = False
                            break

                if match:
                    votes[class_name] = votes.get(class_name, 0) + 1
                    matched_rules += 1

            predicted_class = max(votes, key=votes.get) if votes else "Tidak Berprestasi"
            predictions.append({
                "data_raw": raw_data,  # Nilai mentah ditambahkan ke hasil prediksi
                "data_labeled": row.to_dict(),
                "predicted_class": predicted_class,
                "votes": votes,
                "matched_rules": matched_rules
            })

            # Hitung kategori prediksi
            if predicted_class == "Berprestasi":
                counts["Berprestasi"] += 1
            elif predicted_class == "Cukup Berprestasi":
                counts["Cukup Berprestasi"] += 1
            elif predicted_class == "Tidak Berprestasi":
                counts["Tidak Berprestasi"] += 1

        # Membuat grafik dengan Plotly
        fig = go.Figure(data=[
            go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=["#007bff", "#ffc107", "#dc3545"]
            )
        ])
        fig.update_layout(title="Distribusi Prediksi Prestasi Siswa (20% Data Terakhir)", xaxis_title="Kategori", yaxis_title="Jumlah Siswa")

        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('predict_data_uji.html', predictions=predictions, counts=counts, graph_json=graph_json)

    except Exception as e:
        return render_template('predict_data_uji.html', error=f"Error saat memproses prediksi: {str(e)}")


@app.route('/web')
def web_index():
    """
    Halaman utama web yang menampilkan navigasi ke semua fungsi.
    """
    return render_template('index.html')

@app.route('/web/process')
def web_process():
    """
    Halaman utama web yang menampilkan navigasi ke semua fungsi.
    """
    return render_template('process.html')

@app.route('/web/upload', methods=['GET', 'POST'])
def web_upload():
    """
    Halaman web untuk mengunggah data latih.
    - GET: Menampilkan formulir unggah.
    - POST: Mengunggah file dan mengarahkan kembali ke halaman utama.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part in the request")
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No file selected for uploading")
        if file and file.filename.endswith('.xlsx'):
            file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
            file.save(file_path)
            return render_template('upload.html', success="File successfully uploaded")
        else:
            return render_template('upload.html', error="Allowed file type is .xlsx")
    return render_template('upload.html')


@app.route('/web/data', methods=['GET'])
def web_data():
    """
    Halaman web untuk menampilkan preview data latih.
    """
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    if not os.path.exists(file_path):
        return render_template('data.html', error="Data latih belum diunggah.")
    try:
        data = pd.read_excel(file_path)
        data_preview = data.to_dict(orient='records')
        return render_template('data.html', data_preview=data_preview)
    except Exception as e:
        return render_template('data.html', error=f"Failed to read data latih: {str(e)}")


@app.route('/web/training', methods=['GET'])
def web_training():
    """
    Halaman web untuk memulai pelatihan model.
    """
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    if not os.path.exists(file_path):
        return render_template('training.html', error="Data latih belum diunggah.")
    try:
        # Panggil fungsi training yang sudah ada
        response = training()
        if response[1] == 200:
            return render_template('training.html', success="Model berhasil dilatih dan disimpan.")
        else:
            return render_template('training.html', error=response[0].json.get('error'))
    except Exception as e:
        return render_template('training.html', error=f"Error saat melatih model: {str(e)}")


@app.route('/web/evaluating', methods=['GET'])
def web_evaluating():
    """
    Halaman web untuk menampilkan evaluasi model.
    """
    # Path ke file data latih dan model
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    model_files = [
        os.path.join(MODEL_DIR, "random_forest_model_max_depth.pkl"),
        os.path.join(MODEL_DIR, "encoded_feature_names.pkl"),
        os.path.join(MODEL_DIR, "target_encoder.pkl")
    ]
    
    # Cek apakah file model dan data latih ada
    if not all(os.path.exists(f) for f in model_files) or not os.path.exists(file_path):
        return render_template('evaluating.html', error="Model atau data tidak ditemukan. Silakan latih model terlebih dahulu.")
    
    try:
        # Panggil fungsi evaluating yang sudah ada
        response = evaluating()
        
        # Cek status kode respons
        if response[1] == 200:
            evaluation = response[0].get_json()  # Gunakan get_json() untuk mendapatkan data JSON
            
            # Konversi confusion matrix ke format yang lebih mudah dibaca di HTML
            confusion_matrix = evaluation.get("confusion_matrix", {})
            confusion_matrix_table = []
            classes = sorted(confusion_matrix.keys(), key=lambda x: int(x))
            for true_class in classes:
                row = {"class": true_class, "values": []}
                for pred_class in classes:
                    row["values"].append(confusion_matrix[true_class].get(pred_class, 0))
                confusion_matrix_table.append(row)
            
            # Tambahkan confusion matrix yang telah diformat ke hasil evaluasi
            evaluation["confusion_matrix_table"] = confusion_matrix_table
            
            return render_template('evaluating.html', evaluation=evaluation)
        else:
            # Jika ada error, tampilkan pesan error
            error_message = response[0].get_json().get('error', 'Terjadi kesalahan saat evaluasi model.')
            return render_template('evaluating.html', error=error_message)
    
    except Exception as e:
        # Tangani exception dan tampilkan pesan error
        return render_template('evaluating.html', error=f"Error saat evaluasi model: {str(e)}")
@app.route('/evaluating', methods=['GET'])
def evaluating():
    """
    Endpoint untuk mengevaluasi model menggunakan data uji.
    """
    # Memuat file data uji
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    if not os.path.exists(file_path):
        return jsonify({"error": "Data uji belum diunggah."}), 400

    try:
        # Membaca data uji
        data = pd.read_excel(file_path)

        # Preprocessing Data
        data_cleaned = data.drop(columns=['NO', 'NAMA SISWA', 'NISN', 'NIS'])

        # Encoding variabel kategori 'Status Prestasi' dengan LabelEncoder
        target_encoder_filename = os.path.join(MODEL_DIR, "target_encoder.pkl")
        label_encoder = joblib.load(target_encoder_filename)
        data_cleaned['Status Prestasi'] = label_encoder.transform(data_cleaned['Status Prestasi'])

        # One-Hot Encoding untuk fitur kategorikal
        categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
        data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features)

        # Memastikan fitur sesuai dengan model
        encoder_filename = os.path.join(MODEL_DIR, "encoded_feature_names.pkl")
        encoded_feature_names = joblib.load(encoder_filename)

        # Menambahkan fitur yang hilang dengan nilai default 0
        for feature in encoded_feature_names:
            if feature not in data_encoded.columns:
                data_encoded[feature] = 0

        # Menghapus fitur tambahan yang tidak diharapkan
        data_encoded = data_encoded[encoded_feature_names + ['Status Prestasi']]

        # Pisahkan fitur dan target
        X = data_encoded.drop(columns=['Status Prestasi'])
        y = data_encoded['Status Prestasi']

        # Memuat model dari file
        model_filename = os.path.join(MODEL_DIR, "random_forest_model_max_depth.pkl")
        rf_model = joblib.load(model_filename)

        # Prediksi menggunakan model
        y_pred = rf_model.predict(X)

        # Evaluasi manual
        def calculate_metrics(y_true, y_pred):
            """
            Menghitung akurasi, presisi, recall, dan F1-score secara manual.
            """
            classes = sorted(set(y_true) | set(y_pred))
            confusion_matrix = {true_class: {pred_class: 0 for pred_class in classes} for true_class in classes}

            for true, pred in zip(y_true, y_pred):
                confusion_matrix[true][pred] += 1

            metrics = {}
            total_samples = len(y_true)
            correct_predictions = 0

            for class_label in classes:
                TP = confusion_matrix[class_label][class_label]
                FP = sum(confusion_matrix[other_class][class_label] for other_class in classes if other_class != class_label)
                FN = sum(confusion_matrix[class_label][other_class] for other_class in classes if other_class != class_label)

                precision = TP / (TP + FP) if (TP + FP) != 0 else 0
                recall = TP / (TP + FN) if (TP + FN) != 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

                metrics[class_label] = {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                    "support": TP + FN
                }

                correct_predictions += TP

            accuracy = correct_predictions / total_samples if total_samples > 0 else 0
            return {
                "accuracy": accuracy,
                "confusion_matrix": confusion_matrix,
                "classification_report": metrics
            }

        evaluation = calculate_metrics(y, y_pred)

        # Konversi hasil evaluasi ke format JSON
        evaluation_result = {
            "accuracy": evaluation["accuracy"],
            "confusion_matrix": evaluation["confusion_matrix"],
            "classification_report": evaluation["classification_report"]
        }

        return jsonify(evaluation_result), 200

    except Exception as e:
        return jsonify({"error": f"Error saat evaluasi model: {str(e)}"}), 500
    
@app.route('/web/rules', methods=['GET'])
def web_rules():
    """
    Halaman web untuk menampilkan aturan yang telah diekstrak.
    """
    rules_text_filename = os.path.join(TREE_VIS_DIR, 'rules_readable.txt')
    rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')
    if not os.path.exists(rules_text_filename) or not os.path.exists(rules_json_filename):
        return render_template('rules.html', error="Rules belum tersedia. Silakan latih model terlebih dahulu.")
    try:
        with open(rules_text_filename, 'r') as f:
            rules_text = f.read()
        with open(rules_json_filename, 'r') as f:
            rules_json = json.load(f)
        return render_template('rules.html', rules_text=rules_text, rules_json=rules_json)
    except Exception as e:
        return render_template('rules.html', error=f"Failed to read rules: {str(e)}")


@app.route('/web/visualizations', methods=['GET'])
def web_visualizations():
    """
    Halaman web untuk menampilkan visualisasi pohon keputusan.
    """
    try:
        images = [img for img in os.listdir(STATIC_TREE_VIS_DIR) if img.endswith('.png')]
        image_urls = [url_for('tree_visualization', filename=img, _external=True) for img in images]
        if not images:
            return render_template('visualizations.html', message="Visualisasi pohon belum tersedia. Silakan latih model terlebih dahulu.")
        return render_template('visualizations.html', image_urls=image_urls)
    except Exception as e:
        return render_template('visualizations.html', error=f"Error saat memuat visualisasi: {str(e)}")


@app.route('/web/predict', methods=['GET', 'POST'])
def web_predict():
    """
    Halaman web untuk melakukan prediksi menggunakan aturan.
    - GET: Menampilkan formulir prediksi.
    - POST: Mengirim data untuk diprediksi dan menampilkan hasil.
    """
    if request.method == 'POST':
        try:
            input_data = request.form.to_dict()
            # Normalisasi input_data: ubah semua key dan value menjadi huruf kecil dan hapus spasi
            normalized_input = {k.strip().lower(): v.strip().lower() for k, v in input_data.items()}
    
            # Memuat aturan dari JSON
            rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')
            if not os.path.exists(rules_json_filename):
                return render_template('predict.html', error="Rules tidak ditemukan. Silakan latih model terlebih dahulu.")
            with open(rules_json_filename, 'r') as f:
                rules = json.load(f)
    
            votes = {}
            matched_rules = 0  # Untuk melacak jumlah aturan yang cocok
    
            for rule in rules:
                conditions = rule['conditions']
                class_name = rule['class']
                match = True
                for condition in conditions:
                    # Condition format: "Fitur = "Kategori"" atau "Fitur != "Kategori""
                    if " = " in condition:
                        feature, _, value = condition.partition(" = ")
                        feature = feature.strip().lower()
                        value = value.strip('"').lower()
                        if feature not in normalized_input or normalized_input[feature] != value:
                            match = False
                            break
                    elif " != " in condition:
                        feature, _, value = condition.partition(" != ")
                        feature = feature.strip().lower()
                        value = value.strip('"').lower()
                        if feature not in normalized_input or normalized_input[feature] == value:
                            match = False
                            break
                    else:
                        # Format kondisi tidak didukung
                        match = False
                        break
                if match:
                    votes[class_name] = votes.get(class_name, 0) + 1
                    matched_rules += 1
    
            if matched_rules == 0:
                return render_template('predict.html', error="No rules matched the input data.")
    
            # Tentukan kelas dengan suara terbanyak
            predicted_class = max(votes, key=votes.get)
    
            return render_template('predict.html', predicted_class=predicted_class, votes=votes)
    
        except Exception as e:
            return render_template('predict.html', error=f"Error during prediction: {str(e)}")
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
