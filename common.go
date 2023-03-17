package cnn

import (
	"bytes"
	"fmt"
	"math"
)

func Max(outs ...float64) float64 {
	var max float64

	for _, out := range outs {
		max = math.Max(max, out)
	}

	return max
}

func ArgMax(outs ...float64) int {
	var idx int
	var max float64

	for i, out := range outs {
		if out > max {
			idx = i
			max = out
		}
	}

	return idx
}

func DataInspect(data ...float64) string {
	var buf bytes.Buffer
	for _, item := range data {
		buf.WriteString(fmt.Sprintf("%.06f ", item))
	}
	return buf.String()
}
