#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RNN Decoder module for image captioning.
This module implements the decoder part of the image captioning system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    """
    RNN Decoder for generating captions from image features.
    Uses LSTM/GRU with word embeddings to generate captions word by word.
    """
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, rnn_type='lstm', dropout=0.5):
        """
        Initialize the decoder.
        
        Args:
            embed_size (int): Dimensionality of the input embeddings (from the encoder)
            hidden_size (int): Dimensionality of the RNN hidden state
            vocab_size (int): Size of the vocabulary
            num_layers (int): Number of layers in the RNN
            rnn_type (str): Type of RNN cell ('lstm' or 'gru')
            dropout (float): Dropout probability
        """
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        # TODO: Create the word embedding layer to convert word indices to vectors

        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # TODO: Create the RNN layer (LSTM or GRU) based on rnn_type

        # 1. Check the rnn_type ('lstm' or 'gru')
        # 2. Create the appropriate RNN layer with the specified parameters
        rnn_dropout = dropout if num_layers > 1 else 0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)

        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
            # 3. Handle the case of an unsupported RNN type
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}. Use 'lstm' or 'gru'.")
        
        
        # TODO: Create the output projection layer from hidden_size to vocab_size

        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # TODO: Create a dropout layer with the specified dropout probability

        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions, hidden=None):
        """
        Forward pass for training with teacher forcing.
        
        Args:
            features (torch.Tensor): Image features from the encoder [batch_size, embed_size]
            captions (torch.Tensor): Ground truth captions [batch_size, seq_length]
            hidden (tuple or torch.Tensor, optional): Initial hidden state for the RNN
            
        Returns:
            torch.Tensor: Raw output scores for each word in the vocabulary
                        Shape: [batch_size, seq_length, vocab_size]
            tuple or torch.Tensor: Final hidden state of the RNN
        """
        
        # TODO: Implement the forward pass with teacher forcing

        # 1. Embed the input captions
        embeddings = self.embedding(captions[:, :-1])

        # 2. Prepare image features (add sequence dimension)
        features = features.unsqueeze(1)

        # 3. Concatenate image features with embedded captions
        inputs = torch.cat((features, embeddings), dim=1)

        # 4. Run the RNN on the combined inputs
        rnn_outputs, hidden = self.rnn(inputs, hidden)

        # 5. Apply dropout to the RNN outputs
        rnn_outputs_dropped = self.dropout(rnn_outputs)

        # 6. Project the outputs to vocabulary size
        outputs = self.fc(rnn_outputs_dropped)
        
        return outputs, hidden
    
    def sample(self, features, max_length=20, start_token=1, end_token=2, temperature=1.0, beam_size=1):
        """
        Sample captions using either greedy search or beam search.
        
        Args:
            features (torch.Tensor): Image features from the encoder [batch_size, embed_size]
            max_length (int): Maximum caption length
            start_token (int): Index of the start token
            end_token (int): Index of the end token
            temperature (float): Sampling temperature (higher = more diverse outputs)
            beam_size (int): Beam size for beam search (1 = greedy search)
            
        Returns:
            list: List of generated caption token sequences
        """
        batch_size = features.size(0)
        device = features.device
        
        # If beam size is 1, use greedy sampling
        if beam_size == 1:
            return self._greedy_sample(features, max_length, start_token, end_token, temperature)
        else:
            return self._beam_search(features, max_length, start_token, end_token, beam_size)
    
    def _greedy_sample(self, features, max_length, start_token, end_token, temperature):
        """
        Greedy sampling (beam size = 1)
        """
        batch_size = features.size(0)
        device = features.device
        
        # TODO: Implement greedy sampling for caption generation

        

        # 1. Initialize inputs with start token
        inputs = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)

        # 2. Initialize hidden state (may need to handle LSTM and GRU differently)
        if self.rnn_type == 'lstm':
            h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden = (h0, c0)
        else:
            hidden = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # 3. Create a list to store sampled token indices
        sampled_ids = []
        # 4. Loop for max_length steps:
        for _ in range(max_length):
            #    a. Embed the current input
            embeddings = self.embedding(inputs)

            #    b. Run a single step of the RNN
            rnn_output, hidden = self.rnn(embeddings, hidden)

            #    c. Project to vocabulary size and apply temperature
            output = self.fc(rnn_output.squeeze(1)) 
            output = output / temperature

            #    d. Select the most likely next word
            probabilities = F.softmax(output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)

            #    e. Append to sampled indices
            sampled_ids.append(predicted.unsqueeze(1))

            #    f. Update input for next step
            inputs = predicted.unsqueeze(1)

            #    g. Break if all sequences generated end token
            if (predicted == end_token).all(): break

        # 5. Stack sampled indices into a tensor
        sampled_ids = torch.cat(sampled_ids, dim=1)
        
        return sampled_ids
    
    def _beam_search(self, features, max_length, start_token, end_token, beam_size):
        """
        Beam search sampling for better caption quality.
        
        Note: This implementation is for batch size = 1 for simplicity.
        """
        device = features.device
        
        # We only support batch size 1 for beam search for simplicity
        if features.size(0) != 1:
            raise ValueError("Beam search currently only supports batch size 1")
        
        # Initialize with start token
        k = beam_size
        sequences = [([start_token], 0.0, None)]  # (sequence, score, hidden)
        
        # For the first step, use image features as initial hidden state
        if self.rnn_type == 'lstm':
            # For LSTM, we need to initialize (h0, c0)
            h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
            c0 = torch.zeros_like(h0)
            hidden_init = (h0, c0)
        else:
            # For GRU, we just need to initialize h0
            hidden_init = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        
        # Run beam search
        for _ in range(max_length):
            all_candidates = []
            
            # Expand each current candidate
            for seq, score, hidden in sequences:
                # If sequence ended, keep it
                if seq[-1] == end_token:
                    all_candidates.append((seq, score, hidden))
                    continue
                
                # Forward pass through the model
                inputs = torch.LongTensor([seq[-1]]).unsqueeze(0).to(device)
                embed = self.embedding(inputs)
                
                # Initialize hidden state with image features for the first step
                if len(seq) == 1 and hidden is None:
                    output, hidden_next = self.rnn(embed, hidden_init)
                else:
                    output, hidden_next = self.rnn(embed, hidden)
                
                # Project to vocabulary
                output = self.fc(output.squeeze(1))  # [1, vocab_size]
                
                # Convert to probabilities
                output = F.log_softmax(output, dim=1)
                
                # Get top k candidates
                topk_probs, topk_indices = output.topk(k)
                
                # Create new candidates
                for i in range(k):
                    next_token = topk_indices[0, i].item()
                    next_score = score + topk_probs[0, i].item()
                    next_seq = seq + [next_token]
                    all_candidates.append((next_seq, next_score, hidden_next))
            
            # Select k best candidates
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:k]
            
            # Check if all sequences have ended
            if all(seq[-1] == end_token for seq, _, _ in sequences):
                break
        
        # Return the highest scoring sequence
        best_seq = sequences[0][0]
        return [torch.LongTensor(best_seq).to(device)]