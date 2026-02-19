
import subprocess
import sys
import os

# List of models to run
MODELS = [
    'resnet50',
    'densenet121',
    'convnext_tiny',
    'swin_t',
    'efficientnet_b0',
    'efficientnet_v2b2',
    'efficientnet_v2b3',
    'efficientnet_v2s'
]

def run_command(command, log_file=None):
    """Run a shell command and stream output."""
    print(f"Running: {' '.join(command)}")
    
    # Simple run
    try:
        if log_file:
            with open(log_file, 'w') as f:
                subprocess.run(command, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with exit code {e.returncode}")
        # Optionally exit or continue?
        # sys.exit(1)
        return False
    return True

def main():
    print("="*60)
    print(f"AUTOMATED EXPERIMENT RUNNER")
    print(f"Models to run: {', '.join(MODELS)}")
    print("="*60)
    
    for model in MODELS:
        print(f"\n\n{'='*40}")
        print(f"🚀 PROCESSING MODEL: {model}")
        print(f"{'='*40}")
        
        # 1. Train (skip if model already exists)
        model_path = os.path.join('results', model, 'best_model.pth')
        if os.path.exists(model_path):
            print(f"\n--- Skipping training for {model} (model already exists) ---")
        else:
            print(f"\n--- Training {model} ---")
            train_cmd = [sys.executable, 'train.py', '--backbone', model]
            if not run_command(train_cmd):
                print(f"⚠️ Skipping evaluation for {model} due to training failure.")
                continue
            
        # 2. Evaluate
        results_file = os.path.join('results', model, 'evaluation_results.txt')
        if os.path.exists(results_file):
            print(f"\n--- Skipping evaluation for {model} (results already exist) ---")
        else:
            print(f"\n--- Evaluating {model} ---")
            eval_cmd = [sys.executable, 'evaluate_final.py', '--backbone', model, '--no-confirm']
            run_command(eval_cmd)
        
        # 3. Visualize (generates prediction_samples.png, confusion_matrices.png, detailed_prediction_grid.png)
        viz_file = os.path.join('results', model, 'prediction_samples.png')
        if os.path.exists(viz_file):
            print(f"\n--- Skipping visualization for {model} (visualizations already exist) ---")
        else:
            print(f"\n--- Visualizing {model} ---")
            viz_cmd = [sys.executable, 'visualize_predictions.py', '--backbone', model]
            run_command(viz_cmd)
        
        print(f"✅ Completed {model}")

    print("\n\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()
