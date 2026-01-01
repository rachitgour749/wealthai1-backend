
import os
import shutil
import glob

SOURCE_DIR = r"d:\WEALTHAI_PROD\WEALTH_AI_BACKEND\ChatAI"
DEST_DIR = r"d:\WEALTHAI_PROD\WEALTH_AI_BACKEND\Services\ChatAI"
SRC_SUBDIR = os.path.join(SOURCE_DIR, "src")
DATA_SUBDIR = os.path.join(SOURCE_DIR, "data")

def refactor_chatai():
    # 1. Create Destination
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created {DEST_DIR}")

    # 2. Move Folders from src (api, core, stores, sync) -> Services/ChatAI/
    subdirs = ["api", "core", "stores", "sync"]
    for sub in subdirs:
        src_path = os.path.join(SRC_SUBDIR, sub)
        dst_path = os.path.join(DEST_DIR, sub)
        if os.path.exists(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.move(src_path, dst_path)
            print(f"Moved {sub} to {dst_path}")
        else:
            print(f"Warning: {sub} not found in {SRC_SUBDIR}")

    # 3. Move Data Folder
    if os.path.exists(DATA_SUBDIR):
        dst_data = os.path.join(DEST_DIR, "data")
        if os.path.exists(dst_data):
            shutil.rmtree(dst_data)
        shutil.move(DATA_SUBDIR, dst_data)
        print(f"Moved data to {dst_data}")

    # 4. Move .env
    env_src = os.path.join(SOURCE_DIR, ".env")
    env_dst = os.path.join(DEST_DIR, ".env")
    if os.path.exists(env_src):
        shutil.move(env_src, env_dst)
        print("Moved .env")
    
    # 5. Config/Other files? (requirements.txt is already merged manually via tools)
    
    # 6. Update Imports in all .py files in DEST_DIR
    print("Updating imports...")
    py_files = glob.glob(os.path.join(DEST_DIR, "**", "*.py"), recursive=True)
    
    replacements = [
        ("from src.", "from Services.ChatAI."),
        ("import src.", "import Services.ChatAI.")
    ]
    
    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = content
        for old, new in replacements:
            new_content = new_content.replace(old, new)
        
        if new_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            print(f"Updated imports in {os.path.basename(file_path)}")

    # 7. Create __init__.py in Services/ChatAI if missing
    init_path = os.path.join(DEST_DIR, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("")
        print("Created __init__.py")

if __name__ == "__main__":
    refactor_chatai()
