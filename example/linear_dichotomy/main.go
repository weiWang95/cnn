package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strconv"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/weiWang95/cnn"
)

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 0)
	return v
}

func main() {
	// k := &cnn.Kernel{
	// 	Step: 1,
	// 	Size: 3,
	// 	Value: []float64{
	// 		-1, -1, -1,
	// 		-1, 8, -1,
	// 		-1, -1, -1,
	// 	},
	// }

	// p := &cnn.MaxPooling{
	// 	BasePooling: cnn.BasePooling{Size: 2},
	// }

	// data := [][]float64{
	// 	{0, 1, 0, 0, 0, 0, 0, 0},
	// 	{0, 1, 0, 0, 1, 1, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// 	{0, 1, 0, 0, 0, 0, 1, 0},
	// }

	// r := k.ExtractAll(data)

	// for _, v := range r {
	// 	fmt.Println(v)
	// }

	// r2 := p.PoolingAll(p, r)
	// for _, v := range r2 {
	// 	fmt.Println(v)
	// }

	sprint := 1
	batchSize := 100
	study := 0.1
	minLoss := 0.05
	loss := &cnn.LogisticDiff{}

	l1 := cnn.NewNeuronLayer(1, 2, nil)
	// n := cnn.NewNeuronLayer(2, 2, l1, cnn.WithNLRandomWeight(true), cnn.WithNLActive(&cnn.Sigmoid{}), cnn.WithNLWeigth(0))
	// n = cnn.NewNeuronLayer(2, 4, n, cnn.WithNLRandomWeight(true), cnn.WithNLActive(&cnn.Sigmoid{}))
	end := cnn.NewNeuronLayer(3, 1, l1, cnn.WithNLRandomWeight(true), cnn.WithNLActive(&cnn.Sigmoid{}), cnn.WithNLWeigth(0))
	fmt.Println(sprint, study, end)

	str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"1-0":-6.968860294171455,"1-1":6.854333381944804,"weight":-0.09639785805029938}]]`
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"1-0":-4.52341097124447,"1-1":4.943798390639264,"weight":0.0507297932148732}]]` // 0.24068127934774425
	var ss [][]map[string]float64
	err := json.Unmarshal([]byte(str), &ss)
	if err != nil {
		panic(err)
	}
	cnn.ApplyWeight(l1, ss)

	result := make([]float64, 0)

	// NewPointChart(data)

	times := 0
	for i := 0; i < sprint; i++ {
		data := make([][]float64, 0)
		for {
			x, y := randPoint(1)
			if x-y < 0.05 && x-y > -0.05 {
				continue
			}
			if y > x {
				data = append(data, []float64{x, y, 1})
			} else {
				data = append(data, []float64{x, y, 0})
			}
			if len(data) == batchSize {
				break
			}
		}

		// NewPointChart(data)

		var avgLoss float64
		var success bool
		for _, d := range data {
			out := cnn.Compute(l1, d[0], d[1])
			fmt.Printf("x:%.02f, y:%.02f, A: %.02f[%.02f]\n", d[0], d[1], out[0], d[2])
			lo := loss.Loss(out, []float64{d[2]})
			avgLoss += lo

			if lo < minLoss {
				times += 1
			} else {
				times = 0
			}
			if times > 20 {
				success = true
				break
			}

			// cnn.BP(end, study, loss, d[2])
			result = append(result, lo)
		}
		if success {
			break
		}

		avgLoss = avgLoss / float64(len(data))
		fmt.Printf("avg loss : %v\n", avgLoss)

		// result = append(result, avgLoss)
	}

	// out := cnn.Compute(l1, 0.05, 0.10)
	// fmt.Printf("A: %.02f[%.02f] B: %.02f[%.02f]\n", out[0], 0.01, out[1], 0.99)

	// cnn.BP(end, study, 0.01, 0.99)

	ws := cnn.ExportWeight(l1)
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	NewLineChart(result)
}

func NewLineChart(data []float64) {
	if len(data) == 0 {
		return
	}
	series := make([]chart.Series, 0)
	xValue := make([]float64, 0, len(data))
	for i, _ := range data {
		xValue = append(xValue, float64(i))
	}

	series = append(series, chart.ContinuousSeries{
		Name:    "text",
		XValues: xValue,
		YValues: data,
	})
	graph := chart.Chart{Series: series}
	f, err := os.Create("test_out.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	err = graph.Render(chart.PNG, f)
	if err != nil {
		panic(err)
	}
}

func NewPointChart(data [][]float64) {
	series := make([]chart.Series, 0)
	x1Value := make([]float64, 0)
	x2Value := make([]float64, 0)
	y1Value := make([]float64, 0)
	y2Value := make([]float64, 0)

	for i, _ := range data {
		if data[i][2] == 1 {
			x1Value = append(x1Value, data[i][0])
			y1Value = append(y1Value, data[i][1])
		} else {
			x2Value = append(x2Value, data[i][0])
			y2Value = append(y2Value, data[i][1])
		}
	}

	series = append(series, chart.ContinuousSeries{
		Name: "red",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    5,
			DotColor:    chart.ColorRed,
		},
		XValues: x1Value,
		YValues: y1Value,
	})
	series = append(series, chart.ContinuousSeries{
		Name: "blue",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    5,
			DotColor:    chart.ColorBlue,
		},
		XValues: x2Value,
		YValues: y2Value,
	})
	graph := chart.Chart{Series: series}
	f, err := os.Create("data.png")
	if err != nil {
		panic(err)
	}
	defer f.Close()
	err = graph.Render(chart.PNG, f)
	if err != nil {
		panic(err)
	}
}
