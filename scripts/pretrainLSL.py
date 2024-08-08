import torch
import os
from fairseq import options
from models.custom_transformer import CustomTransformerEncoder
from scripts.utils import setup_data, setup_optimizer

def main():
    args = options.parse_args_and_arch()
    dictionary, embed_tokens, train_data = setup_data(args)
    
    model = CustomTransformerEncoder(args, dictionary, embed_tokens)
    optimizer = setup_optimizer(args, model)
    
    # Training loop
    for epoch in range(args.max_epoch):
        for batch in train_data:
            optimizer.zero_grad()
            output = model(batch.src_tokens, batch.src_lengths, lang_pair=batch.lang_pair)
            loss = model.compute_loss(output, batch)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'checkpoint_{epoch}.pt'))

    # Save the learned roles (e.g., mixing weights)
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer_roles': model.get_layer_roles(),  # Assuming you have a method to get learned roles
    }, os.path.join(args.save_dir, 'pretrained_model_with_roles.pt'))

if __name__ == "__main__":
    main()
