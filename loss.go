package cnn

import "math"

type ILoss interface {
	Loss(outs []float64, expects []float64) float64
	BP(out float64, expect float64) float64
}

var (
	SquareDiff   ILoss = &squareDiff{}
	LogisticDiff ILoss = &logisticDiff{}
)

type squareDiff struct{}

func (d *squareDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		loss += 0.5 * math.Pow(expects[i]-outs[i], 2)
	}

	return loss
}
func (d *squareDiff) BP(out float64, expect float64) float64 {
	return -(expect - out)
}

type logisticDiff struct{}

func (d *logisticDiff) Loss(outs []float64, expects []float64) float64 {
	var loss float64

	for i, _ := range outs {
		out := outs[i]
		expect := expects[i]
		loss += -1 * (expect*math.Log(out) + (1-expect)*math.Log(1-out))
	}

	return loss
}

func (d *logisticDiff) BP(out float64, expect float64) float64 {
	return (out - expect) / (out * (1 - out))
}
