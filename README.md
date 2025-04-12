# Hangman Solver using Transformer MLMs and Heuristics

## Description

This project implements an intelligent Hangman solver that leverages a combination of Transformer-based Masked Language Models (MLMs) and statistical heuristic strategies to predict letters and solve Hangman puzzles. The goal is to maximize the win rate by effectively guessing letters in unknown words.

The approach integrates multiple Transformer models trained for different word characteristics (high, medium, low complexity, and short words) with heuristic methods based on letter frequency and pattern matching. These predictions are weighted empirically based on the game state (e.g., proportion of unknown letters, word length) to determine the best letter to guess next.

## Features

* **Transformer-Based MLMs:** Utilizes multiple Transformer models (High, Medium, Low, Short) tailored for different word complexities and lengths.
* **Heuristic Strategy:** Incorporates letter frequency analysis and pattern matching against a dictionary to refine predictions.
* **Hybrid Approach:** Combines MLM predictions and heuristic scores using empirically determined weights for optimal guessing.
* **Dictionary Management:** Efficiently filters word lists based on the current state of the game.
* **Model Architecture:**
    * Vocabulary Size: 28 (letters + MASK + underscore)
    * Embedding Dimension: 128
    * Transformer Heads: 4
    * Transformer Layers: 6
    * Feedforward Dimension: 512
    * Max Sequence Length: 20 (standard models), 10 (short model)

## Implementation Details

The core logic is implemented in the `trexquant-2.ipynb` notebook. Key components include:

1.  **Dictionary Management:** Builds and filters the word dictionary based on game progress (`build_dictionary`, `build_substring_dictionary`).
2.  **Model Inference:** Loads pre-trained models (`.pt` files) and generates letter probabilities for masked positions (`get_model_probs`).
3.  **Guessing Logic:**
    * Calculates heuristic scores based on the filtered dictionary.
    * Retrieves probabilities from relevant MLMs.
    * Combines MLM probabilities and heuristic scores using dynamic weights based on word length and unknown ratio.
    * Selects the highest-scoring, un-guessed letter. Includes a fallback strategy.

*(Refer to `Trexquant_Hangman_Srijan.pdf` and `trexquant-2.ipynb` for detailed code and explanations)*

## Models

Four distinct Transformer MLM models are trained and used:

* `hangman_mlm_high.pt`: For complex words.
* `hangman_mlm_medium.pt`: For words of medium complexity.
* `hangman_mlm_low.pt`: For simpler words.
* `hangman_mlm_short.pt`: Optimized for words with length <= 7.

These models are combined with a heuristic approach, and their contributions are weighted based on game context (unknown letter ratio, word length).

## Results

* **Training Win Rate:** ~60%
* **Final Execution Win Rate:** 56.8% (Slight decrease attributed to late-stage weight adjustments without sufficient validation)
* **Training Loss:** Graphs showing model convergence are available in the `Trexquant_Hangman_Srijan.pdf` document.

## Future Improvements

* **Computational Resources:** Training deeper and more complex models with more epochs could improve accuracy, but was limited by available hardware (MacBook M2 Air, Kaggle GPUs).
* **Weight Optimization:** Further refinement and validation of the weighting strategy for combining model and heuristic predictions.

## Acknowledgements

* Heuristic approach inspired by insights from Shashank Kartikey's GitHub repository.
* Developed as part of the Trexquant Hangman Challenge.
