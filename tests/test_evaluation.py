import pytest
import torch
from evaluate_simCLR import accuracy

def test_accuracy()-> None:
    # Create sample data
    output = torch.tensor([[0.1, 0.5, 0.3], [0.2, 0.6, 0.2]])
    target = torch.tensor([1, 2])
    topk = (1,)

    # Calculate accuracy
    res = accuracy(output, target, topk=topk)

    # Check if the result matches the expected accuracy
    expected_accuracy = [50.0]
    assert res == expected_accuracy

def test_accuracy_topk_5():
    # Create sample data
    output = torch.tensor([[0.1, 0.5, 0.3, 0.1, 0.4, 0.5, 0.2, 0.3, 0.1, 0.9], 
                           [0.2, 0.6, 0.2, 0.1, 0.3, 0.6, 0.2, 0.4, 0.1, 0.8],
                           [0.3, 0.4, 0.3, 0.2, 0.5, 0.3, 0.1, 0.7, 0.2, 0.6],
                           [0.4, 0.3, 0.3, 0.5, 0.6, 0.1, 0.2, 0.8, 0.1, 0.7]])
    target = torch.tensor([6, 7, 8, 9])  # Targets that are not in the top 5
    topk = (5,)

    # Calculate accuracy
    res = accuracy(output, target, topk=topk)
    print(res)

    # Check if the result matches the expected accuracy
    # In this case, the expected accuracy is 25.0 for all samples
    expected_accuracy = [50.0]
    assert res == expected_accuracy 