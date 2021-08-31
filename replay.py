import os
import glob

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def createReplayVideo(replayfile):
    """Call the script from the retro library that produces .mp4 replays from the saved .bk2 files of the environment"""
    os.system(f"python3 -m retro.scripts.playback_movie {replayfile}")

def getReplayFiles(dir):
    """Get all absolute .bk2 filenames in a given directory"""
    return glob.glob(f"{dir}/*.bk2")

def createReplayVideosFromDir(dir):
    """Gets all .bk2 files from directory and creates the corresponding .mp4 in the same directory"""
    rfiles = getReplayFiles(dir)
    for rfile in rfiles:
        createReplayVideo(rfile)

if __name__ == '__main__':
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/my_ppo_health")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/my_ppo_health_2")

    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/my_ppo_score")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/my_ppo_score_2")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/my_ppo_score_3")

    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/ppo_health")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/ppo_score")

    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/a2c_health")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/a2c_score")

    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/dqn_health")
    createReplayVideosFromDir(f"{BASE_DIR}/evaluation/dqn_score")
