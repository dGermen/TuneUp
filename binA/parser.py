import argparse
import os

# Create the parser
parser = argparse.ArgumentParser(description='TUI parser')

# Add arguments
parser.add_argument('TMN', help='Trained model name')
parser.add_argument('mode', help='Wanted mode')
parser.add_argument('--cold_start_ratio', type=float, default=0.0, help='Cold start ratio')
parser.add_argument('--training_type', choices=['base', 'tuneup', 'wo_syn', 'wo_cur'], default='base', help='Training type')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values
print('TMN:', args.TMN)
print('mode:', args.mode)
print('cold_start_ratio:', args.cold_start_ratio)
print('training_type:', args.training_type)


# Check models folder for TMN
models_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
if args.TMN is None:
    # Train a new model
    print('Training a new model...')
     
else:
    # Load and test the model
    model_path = os.path.join(models_folder, args.TMN)
    if os.path.exists(model_path):
        print('Loading and testing the model:', args.TMN)
        # Your code for loading and testing the model goes here
    else:
        print('Model not found:', args.TMN)