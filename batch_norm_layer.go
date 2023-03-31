package cnn

import "math"

type BatchNormLayer struct {
	gamma        []float64
	beta         []float64
	eps          float64
	runningMean  []float64
	runningVar   []float64
	alpha        float64
	learningRate float64
}

func NewBatchNormLayer(shape []int, eps float64) *BatchNormLayer {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	return &BatchNormLayer{
		gamma:        make([]float64, size),
		beta:         make([]float64, size),
		eps:          eps,
		runningMean:  make([]float64, size),
		runningVar:   make([]float64, size),
		alpha:        0.9,
		learningRate: 0.01,
	}
}

func (bn *BatchNormLayer) Forward(x []float64, train bool) []float64 {
	if train {
		mean := 0.0
		for _, v := range x {
			mean += v
		}
		mean /= float64(len(x))

		variance := 0.0
		for _, v := range x {
			variance += math.Pow(v-mean, 2)
		}
		variance /= float64(len(x))

		stdDev := math.Sqrt(variance + bn.eps)

		bn.runningMean = bn.batchNormRunningAverage(bn.runningMean, mean)
		bn.runningVar = bn.batchNormRunningAverage(bn.runningVar, variance)

		result := make([]float64, len(x))
		for i, v := range x {
			result[i] = bn.gamma[i]*(v-mean)/stdDev + bn.beta[i]
		}

		return result
	} else {
		variance := 0.0
		for _, v := range bn.runningVar {
			variance += v
		}
		variance /= float64(len(bn.runningVar))
		stdDev := math.Sqrt(variance + bn.eps)

		result := make([]float64, len(x))
		for i, v := range x {
			result[i] = bn.gamma[i]*(v-bn.runningMean[i])/stdDev + bn.beta[i]
		}

		return result
	}
}

func (bn *BatchNormLayer) Backward(grads []float64, x []float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))

	variance := 0.0
	for _, v := range x {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(x))

	stdDev := math.Sqrt(variance + bn.eps)

	dGamma := make([]float64, len(bn.gamma))
	dBeta := make([]float64, len(bn.beta))
	dx := make([]float64, len(x))

	for i, v := range x {
		dx[i] = grads[i] * bn.gamma[i] / stdDev
		dGamma[i] = grads[i] * (v - mean) / stdDev
		dBeta[i] += grads[i]
	}

	dGamma = bn.batchNormRunningAverage(dGamma, 0.0)
	dBeta = bn.batchNormRunningAverage(dBeta, 0.0)

	bn.gamma = bn.batchNormUpdateParams(bn.gamma, dGamma)
	bn.beta = bn.batchNormUpdateParams(bn.beta, dBeta)

	dx = batchNormBackward(dx, x)

	return dx
}

func (bn *BatchNormLayer) batchNormRunningAverage(runningAvg []float64, newAvg float64) []float64 {
	for i, v := range runningAvg {
		runningAvg[i] = bn.alpha*v + (1-bn.alpha)*newAvg
	}
	return runningAvg
}

func (bn *BatchNormLayer) batchNormUpdateParams(param []float64, grad []float64) []float64 {
	for i, v := range param {
		param[i] = v - bn.learningRate*grad[i]
	}
	return param
}

func batchNormBackward(grads []float64, x []float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))

	variance := 0.0
	for _, v := range x {
		variance += math.Pow(v-mean, 2)
	}
	variance /= float64(len(x))

	stdDev := math.Sqrt(variance)

	dVar := 0.0
	dMean := 0.0
	dx := make([]float64, len(x))
	for i, v := range x {
		dx[i] = (grads[i] / stdDev)
		dVar += dx[i] * (v - mean) * (-0.5) * math.Pow(stdDev, -3)
		dMean += -dx[i] / stdDev
	}
	dVar /= float64(len(x))
	dMean /= float64(len(x))

	for i := range dx {
		dx[i] += dVar*2*(x[i]-mean)/float64(len(x)) + dMean
	}

	return dx
}
