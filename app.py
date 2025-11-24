# --- app.py の完全なクリーンコード (コメントアウト修正済み) ---
import streamlit as st
import cv2
# ... (他のインポートや関数の定義は省略、全て含める) ...

def process_video_for_analysis(video_path, label):
    # ... (関数の本体は省略) ...
    
    # 修正後の戻り値
    df = pd.DataFrame(data_to_save)
    return df, None # 👈 None を返す

def generate_swing_feedback(df_ref, df_user):
    # ... (関数の本体は省略) ...
    return "\n".join(feedback)


# --- Streamlitアプリの構成 ---

st.set_page_config(page_title="AIスイング分析デモ", layout="wide")
st.title("⚾️ AIスイング分析システム デモアプリ")
st.markdown("### MediaPipe Poseによる客観的なスイング比較")

REF_VIDEO_PATH = 'my_swing.mp4' 

# ... (uploaded_fileの処理) ...

# --- 結果の表示部分 (st.videoはコメントアウト済み) ---
# ...
            st.header("1. 解析動画の比較")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("手本スイング (リファレンス)")
#                st.video(ref_video_path) # 👈 ここはコメントアウト
            
            with col2:
                st.subheader("自分のスイング (解析済み)")
#                st.video(user_video_path) # 👈 ここはコメントアウト

            st.header("2. 自動フィードバック")
            feedback = generate_swing_feedback(df_ref, df_user)
            st.markdown(feedback)
            # ... (残りのグラフ描画コードは省略、全て含める) ...
            
            # ファイル削除のコードはすべてコメントアウト
#            # 一時ファイルのクリーンアップ
#            if os.path.exists(temp_user_path):
#                os.remove(temp_user_path)
#            if os.path.exists(ref_video_path):
#                os.remove(ref_video_path)
#            if os.path.exists(user_video_path):
#                os.remove(user_video_path)