package cnn

import (
	"math"
	"math/rand"
)

type WeightGenerator interface {
	GetWeight(inputSize, outputSize int) [][]float64
}

type xavierGenerator struct{}

func (xavierGenerator) GetWeight(inputSize, outputSize int) [][]float64 {
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64() * math.Sqrt(1/float64(inputSize))
		}
	}
	return weights
}

type heGenerator struct{}

func (heGenerator) GetWeight(inputSize, outputSize int) [][]float64 {
	weights := make([][]float64, inputSize)
	for i := range weights {
		weights[i] = make([]float64, outputSize)
		for j := range weights[i] {
			weights[i][j] = rand.NormFloat64() * math.Sqrt(2/float64(inputSize))
		}
	}
	return weights
}
