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

import plotly
app = Flask(__name__)

# Direktori untuk menyimpan model dan visualisasi
MODEL_DIR = 'models'
TREE_VIS_DIR = 'tree_visualization'
DATA_DIR = 'data'
STATIC_TREE_VIS_DIR = os.path.join('static', 'tree_visualization')

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

        # Membagi Data ke Training dan Testing
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
            "message": "Model berhasil dilatih dan disimpan.",
            "rules_text": "rules_readable.txt",
            "rules_json": "rules.json",
            "visualized_trees": len(trees_to_visualize)
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error saat melatih model: {str(e)}"}), 500

@app.route('/evaluating', methods=['GET'])
def evaluating():
    model_filename = os.path.join(MODEL_DIR, "random_forest_model_max_depth.pkl")
    encoder_filename = os.path.join(MODEL_DIR, "encoded_feature_names.pkl")
    target_encoder_filename = os.path.join(MODEL_DIR, "target_encoder.pkl")
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')

    if not all([os.path.exists(model_filename), os.path.exists(encoder_filename), os.path.exists(target_encoder_filename), os.path.exists(file_path)]):
        return jsonify({"error": "Model atau data tidak ditemukan. Silakan latih model terlebih dahulu."}), 400

    try:
        # Memuat model dan encoders
        rf_model = joblib.load(model_filename)
        encoded_feature_names = joblib.load(encoder_filename)
        label_encoder = joblib.load(target_encoder_filename)

        data = pd.read_excel(file_path)
        data_cleaned = data.drop(columns=['NO', 'NAMA SISWA', 'NISN', 'NIS'])

        # Encoding variabel kategori 'Status Prestasi' dengan LabelEncoder
        data_cleaned['Status Prestasi'] = label_encoder.transform(data_cleaned['Status Prestasi'])

        # One-Hot Encoding untuk fitur kategorikal
        categorical_features = [col for col in data_cleaned.columns if data_cleaned[col].dtype == 'object']
        data_encoded = pd.get_dummies(data_cleaned, columns=categorical_features)

        # Memastikan semua fitur yang di-encode sudah ada
        # Jika ada fitur baru yang tidak ada saat training, tambahkan kolom tersebut dengan nilai 0
        for feature in encoded_feature_names:
            if feature not in data_encoded.columns:
                data_encoded[feature] = 0

        # Pisahkan fitur dan target
        X = data_encoded.drop(columns=['Status Prestasi'])
        y = data_encoded['Status Prestasi']

        # Pastikan urutan fitur sama seperti saat training
        X = X[encoded_feature_names]

        # Membagi Data ke Training dan Testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Evaluasi Model
        y_pred = rf_model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return jsonify({
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }), 200
    except Exception as e:
        return jsonify({"error": f"Error saat evaluasi model: {str(e)}"}), 500

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
    dalam bentuk card dan grafik.
    """
    file_path = os.path.join(DATA_DIR, 'data_uji.xlsx')
    rules_json_filename = os.path.join(TREE_VIS_DIR, 'rules.json')

    if not os.path.exists(file_path):
        return render_template('predict_data_uji.html', error="Data uji belum diunggah.")

    if not os.path.exists(rules_json_filename):
        return render_template('predict_data_uji.html', error="Rules tidak ditemukan. Silakan latih model terlebih dahulu.")

    try:
        # Membaca data uji
        data = pd.read_excel(file_path)
        if data.empty:
            return render_template('predict_data_uji.html', error="Data uji kosong.")

        def label_nilai(value):
            """Mengonversi nilai numerik menjadi kategori label."""
            if value <= 40:
                return "Sangat Buruk"
            elif 40 < value <= 55:
                return "Buruk"
            elif 55 < value <= 75:
                return "Cukup"
            elif 76 < value <= 85:
                return "Baik"
            else:
                return "Sangat Baik"

        # Simpan data mentah sebelum diubah ke label
        data_raw = data.copy()

        # Melabeli data numerik menjadi kategori
        data_labeled = data.copy()
        numeric_columns = data_labeled.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            data_labeled[col] = data_labeled[col].apply(label_nilai)

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
                "index": index + 1,
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
        fig.update_layout(title="Distribusi Prediksi Prestasi Siswa", xaxis_title="Kategori", yaxis_title="Jumlah Siswa")

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
    file_path = os.path.join(DATA_DIR, 'main_data.xlsx')
    model_files = [
        os.path.join(MODEL_DIR, "random_forest_model_max_depth.pkl"),
        os.path.join(MODEL_DIR, "encoded_feature_names.pkl"),
        os.path.join(MODEL_DIR, "target_encoder.pkl")
    ]
    if not all(os.path.exists(f) for f in model_files) or not os.path.exists(file_path):
        return render_template('evaluating.html', error="Model atau data tidak ditemukan. Silakan latih model terlebih dahulu.")
    try:
        # Panggil fungsi evaluating yang sudah ada
        response = evaluating()
        if response[1] == 200:
            evaluation = response[0].json
            return render_template('evaluating.html', evaluation=evaluation)
        else:
            return render_template('evaluating.html', error=response[0].json.get('error'))
    except Exception as e:
        return render_template('evaluating.html', error=f"Error saat evaluasi model: {str(e)}")


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
