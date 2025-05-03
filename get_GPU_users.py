import subprocess
import os
import pwd
from collections import defaultdict

def get_gpu_usage_by_user():
    try:
        # Run nvidia-smi to get GPU UUIDs and PIDs of active processes
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print("Error running nvidia-smi:", result.stderr)
            return {}

        user_gpu_map = defaultdict(set)

        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            try:
                gpu_uuid, pid_str = line.strip().split(',')
                pid = int(pid_str.strip())

                # Get the user ID and resolve to username
                uid = os.stat(f'/proc/{pid}').st_uid
                username = pwd.getpwuid(uid).pw_name

                # Add the GPU UUID to the user's set
                user_gpu_map[username].add(gpu_uuid.strip())
            except Exception:
                continue  # Skip any errors (e.g. process exited)

        return user_gpu_map

    except Exception as e:
        print("Unexpected error:", e)
        return {}

if __name__ == "__main__":
    user_gpu_map = get_gpu_usage_by_user()
    if user_gpu_map:
        print("GPU usage by user:")
        for user, gpus in sorted(user_gpu_map.items()):
            print(f" - {user}: {len(gpus)} GPU(s) in use")
    else:
        print("No GPU usage found.")
