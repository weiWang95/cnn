package cnn

import "math"

type ILoss interface {
	Loss(outs []float64, expects []float64) float64
	BP(out float64, expect float64) float64
}

type SquareDiff struct{}

func (d *SquareDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		loss += 0.5 * math.Pow(expects[i]-outs[i], 2)
	}

	return loss
}
func (d *SquareDiff) BP(out float64, expect float64) float64 {
	return -(expect - out)
}

type LogisticDiff struct{}

func (d *LogisticDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		out := outs[i]
		expect := expects[i]
		loss += -1 * (expect*math.Log(out) + (1-expect)*math.Log(1-out))
	}

	return loss
}

func (d *LogisticDiff) BP(out float64, expect float64) float64 {
	return (out - expect) / (out * (1 - out))
}
