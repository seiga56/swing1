import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import os
import uuid

# --- 1. 定数と初期化 ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- 2. 分析ロジック関数 ---

def calculate_angle(a, b, c):
    """3点 a, b, c を受け取り、b を頂点とする角度（度数）を計算する"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_rotation_angle(p_left, p_right):
    """左右のランドマークから水平線に対する傾き角度を計算する"""
    p_left, p_right = np.array(p_left), np.array(p_right)
    connect_vector = p_right - p_left
    angle_rad = np.arctan2(connect_vector[1], connect_vector[0])
    return np.degrees(angle_rad)

@st.cache_data
def process_video_for_analysis(video_path, label):
    """動画解析、データ抽出、解析動画生成を一括で実行するコア関数"""
    
    st.info(f"⌛ {label} の動画解析中です...")
    
    data_to_save = []
    
    # 一意な名前を付けて一時ファイルを生成
    temp_raw_path = f"temp_raw_{uuid.uuid4()}.mp4"
    temp_analysis_path = f"temp_analysis_{uuid.uuid4()}.mp4"
    
    # 適切なフィルターを適用（手動修正の経験を活かす）
    # Streamlit Cloudでは動画のメタデータが安定しないため、自動回転は行わないことが多い
    FILTERS = 'transpose=2,vflip,hflip' # これまでの失敗から最も可能性の高いパターン
    
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"❌ エラー: 動画ファイル {video_path} を開けません。")
            return None, None

        # 動画書き出し設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 生の解析動画を一時ファイルに書き出し
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        
        frame_num = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success: break
            
            image.flags.writeable = False
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image.flags.writeable = True

            if results.pose_landmarks:
                try:
                    # 肘の角度
                    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    r_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    r_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    r_elbow_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                    # 肩の傾き
                    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_tilt = calculate_rotation_angle(l_shoulder, r_shoulder) 

                    # 腰の傾き
                    l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    hip_tilt = calculate_rotation_angle(l_hip, r_hip) 
                    
                    data_to_save.append({'Frame': frame_num, 'Elbow_Angle': r_elbow_angle, 'Shoulder_Tilt': shoulder_tilt, 'Hip_Tilt': hip_tilt})
                    
                    # 骨格の描画
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except Exception as e:
                    # 検出エラーをスキップ
                    pass
            
            out.write(image)
            frame_num += 1
            
        cap.release()
        out.release()
        
    # ffmpegによる向きの修正と最終出力
    try:
        subprocess.run(['ffmpeg', 
                        '-i', temp_raw_path, 
                        '-vf', FILTERS, # フィルター適用
                        '-c:v', 'libx264', '-crf', '23', '-y', temp_analysis_path], 
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_raw_path) # 一時ファイルを削除
    except subprocess.CalledProcessError as e:
        st.warning(f"⚠️ 動画の向き修正エラー。元の解析動画をそのまま表示します。")
        temp_analysis_path = temp_raw_path

    df = pd.DataFrame(data_to_save)
    return df, temp_analysis_path

def plot_comparison(df_ref, df_user, metric, title):
    """手本と自分のスイングの指標を比較グラフとして描画する"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # データをフレーム数に合わせて適切にプロット
    ax.plot(df_user['Frame'], df_user[metric], label='自分のスイング (青線)', color='blue', linestyle='-')
    ax.plot(df_ref['Frame'], df_ref[metric], label='手本スイング (赤破線)', color='red', linestyle='--')
    
    ax.set_xlabel('フレーム数 (時間経過)')
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def generate_swing_feedback(df_ref, df_user):
    """データに基づいて自動で改善点を指摘する"""
    feedback = ["\n--- 🤖 自動フィードバック（AI分析に基づく客観的評価） ---"]
    
    # 1. 肘のタメ（最小角度）比較
    min_angle_ref = df_ref['Elbow_Angle'].min()
    min_angle_user = df_user['Elbow_Angle'].min()
    diff_elbow = min_angle_user - min_angle_ref
    
    if diff_elbow > 10:
        feedback.append(f"⚠️ **肘のタメ不足の可能性:** インパクト時の最小肘角度が手本より約 {diff_elbow:.1f}度 浅いです。タメを深くすることで、より強いインパクトが期待できます。")
    elif diff_elbow < -10:
        feedback.append(f"✅ 肘のタメが手本よりも深い可能性があります。リリースが遅れていないか確認しましょう。")

    # 2. 肩の傾き（安定性）比較
    std_shoulder_user = df_user['Shoulder_Tilt'].std()
    std_shoulder_ref = df_ref['Shoulder_Tilt'].std()
    
    if std_shoulder_user > std_shoulder_ref * 1.5:
        feedback.append(f"❗️ **肩の軸の不安定性:** 肩の傾き（チルト）の変動が手本より大きく、軸がブレている可能性があります。スイング中の頭の位置を意識しましょう。")
    
    if len(feedback) == 1:
         feedback.append("✨ **現時点では大きな課題は見られません。** グラフで細かいタイミングのズレを確認してください。")

    return "\n".join(feedback)


# --- 3. Streamlitアプリの構成 ---

st.set_page_config(page_title="AIスイング分析デモ", layout="wide")
st.title("⚾️ AIスイング分析システム デモアプリ")
st.markdown("### MediaPipe Poseによる客観的なスイング比較")

# 手本動画は事前にリポジトリにアップロードされているものとする
REF_VIDEO_PATH = 'my_swing.mp4' 

# ユーザー動画のアップロード
st.sidebar.header("ステップ 1: 動画をアップロード")
uploaded_file = st.sidebar.file_uploader("比較したい動画 (自分のスイング) をアップロードしてください", type=["mp4", "mov"])

if uploaded_file is not None:
    # --- 処理開始 ---
    
    # アップロードされたファイルを一時的に保存
    temp_user_path = f"temp_user_{uuid.uuid4()}.mp4"
    with open(temp_user_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.sidebar.success("動画のアップロードが完了しました。")

    # 手本動画とユーザー動画の解析を実行 (キャッシュを使用)
    try:
        df_ref, ref_video_path = process_video_for_analysis(REF_VIDEO_PATH, "手本スイング")
        df_user, user_video_path = process_video_for_analysis(temp_user_path, "自分のスイング")

        if df_ref is not None and df_user is not None:
            
            # --- 結果の表示 ---
            st.header("1. 解析動画の比較")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("手本スイング (リファレンス)")
                st.video(ref_video_path)
            
            with col2:
                st.subheader("自分のスイング (解析済み)")
                st.video(user_video_path)

            st.header("2. 自動フィードバック")
            feedback = generate_swing_feedback(df_ref, df_user)
            st.markdown(feedback)

            st.header("3. 数値データの比較グラフ")
            
            st.subheader("右肘の角度変化 (Elbow Angle)")
            fig_elbow = plot_comparison(df_ref, df_user, 'Elbow_Angle', '右肘の角度変化 (Elbow Angle)')
            st.pyplot(fig_elbow)
            
            st.subheader("肩の傾き変化 (Shoulder Tilt)")
            fig_shoulder = plot_comparison(df_ref, df_user, 'Shoulder_Tilt', '肩の傾き変化 (Shoulder Tilt)')
            st.pyplot(fig_shoulder)
            
            st.subheader("腰の傾き変化 (Hip Tilt)")
            fig_hip = plot_comparison(df_ref, df_user, 'Hip_Tilt', '腰の傾き変化 (Hip Tilt)')
            st.pyplot(fig_hip)

            # 一時ファイルのクリーンアップ
            if os.path.exists(temp_user_path):
                os.remove(temp_user_path)
            if os.path.exists(ref_video_path):
                os.remove(ref_video_path)
            if os.path.exists(user_video_path):
                os.remove(user_video_path)
                
        else:
            st.error("動画解析中にエラーが発生しました。動画ファイルが正しい形式か確認してください。")
            
    except Exception as e:
        st.error(f"予期せぬエラーが発生しました: {e}")

else:
    st.info("左のサイドバーから動画ファイルをアップロードして、AIスイング分析を開始してください。")