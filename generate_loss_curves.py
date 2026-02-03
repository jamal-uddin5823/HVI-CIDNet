import re
import matplotlib.pyplot as plt

# Read the training log
with open('train.log', 'r', encoding='utf-8', errors='ignore') as f:
    log_content = f.read()

# Define the three models and their patterns
model_patterns = [
    ('baseline', r'Starting training: baseline'),
    ('face_loss3', r'Starting training: face_loss3'),
    ('face_loss5', r'Starting training: face_loss5')
]

# Extract loss values for each model (first 40 epochs)
model_losses = {}
model_names = ['baseline', 'face_loss3', 'face_loss5']

for model_name in model_names:
    model_losses[model_name] = []

# Pattern to match epoch loss lines
epoch_pattern = r'===> Epoch\[(\d+)\]: Loss: ([\d.]+)'

# Split by model training starts
sections = []
for i, (model_name, pattern) in enumerate(model_patterns):
    matches = list(re.finditer(pattern, log_content))
    if matches:
        start_pos = matches[0].end()
        # Find end (next model start or end of file)
        end_pos = len(log_content)
        if i < len(model_patterns) - 1:
            next_matches = list(re.finditer(model_patterns[i+1][1], log_content))
            if next_matches:
                end_pos = next_matches[0].start()
        sections.append((model_name, log_content[start_pos:end_pos]))

# Extract losses from each section
for model_name, section in sections:
    epoch_matches = re.findall(epoch_pattern, section)
    losses = []
    for epoch_str, loss_str in epoch_matches:
        epoch_num = int(epoch_str)
        if epoch_num <= 40:
            losses.append(float(loss_str))
    model_losses[model_name] = losses

print("Extracted losses:")
for model_name, losses in model_losses.items():
    print(f"{model_name}: {len(losses)} epochs")

# Create the plot
plt.figure(figsize=(10, 6))

colors = {'baseline': 'blue', 'face_loss3': 'green', 'face_loss5': 'red'}

for model_name, losses in model_losses.items():
    if losses:
        epochs = range(1, len(losses) + 1)
        plt.plot(epochs, losses, label=model_name, color=colors[model_name], linewidth=2)

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curves for 40 Epochs', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('loss_curves_40_epochs.png', dpi=300)
plt.savefig('loss_curves_40_epochs.pdf')
print("\nSaved: loss_curves_40_epochs.png and loss_curves_40_epochs.pdf")

# Also print out the loss values
print("\nLoss values for each model:")
for model_name, losses in model_losses.items():
    print(f"\n{model_name}:")
    for i, loss in enumerate(losses, 1):
        print(f"  Epoch {i}: {loss:.4f}")
