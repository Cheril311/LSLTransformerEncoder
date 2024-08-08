import torch
import os
from fairseq import options
from models.custom_transformer import CustomTransformerEncoder
from scripts.utils import setup_data, setup_optimizer

def main():
    args = options.parse_args_and_arch()
    dictionary, embed_tokens, train_data = setup_data(args)
    
    # Load pre-trained weights and roles
    checkpoint = torch.load(os.path.join(args.save_dir, 'pretrained_model_with_roles.pt'))
    model = CustomTransformerEncoder(args, dictionary, embed_tokens)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Assign the roles learned from pre-training
    args.layer_roles = checkpoint['layer_roles']
    model.assign_roles(args.layer_roles) 
    
    optimizer = setup_optimizer(args, model)
    
    # Fine-tuning loop
    for epoch in range(args.max_epoch):
        for batch in train_data:
            optimizer.zero_grad()
            output = model(batch.src_tokens, batch.src_lengths, lang_pair=batch.lang_pair)
            loss = model.compute_loss(output, batch)
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), os.path.join(args.save_dir, f'posttrained_checkpoint_{epoch}.pt'))

if __name__ == "__main__":
    main()
