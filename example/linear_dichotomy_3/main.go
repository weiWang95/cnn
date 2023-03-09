package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/weiWang95/cnn"
)

type Data struct {
	Inputs  [][]float64 `json:"inputs"`
	Expects [][]float64 `json:"expects"`
}

func main() {
	n := cnn.NewNeuralNetwork([]int64{2, 2, 1}, []cnn.IActive{cnn.LeakyReLU, cnn.Sigmoid}, cnn.LogisticDiff)

	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":0.4,"0-1":0.4,"weight":0.1},{"0-0":-0.4,"0-1":-0.4,"weight":0.1}],[{"1-0":0.4,"1-1":0.2,"weight":0.1}]]`
	str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-1.3321899513787225,"0-1":1.3153118291249737,"weight":-0.0025476508128429768},{"0-0":-4.994017108368555,"0-1":1.0383126734434818,"weight":-1.0506058049185316}],[{"1-0":7.709941276072755,"1-1":7.679521938191185,"weight":-3.3414160550111562}]]` //0.003701116800855907
	var wm cnn.WeightMap
	err := json.Unmarshal([]byte(str), &wm)
	if err != nil {
		panic(err)
	}
	n.ApplyWeight(wm)

	// data := generateDataV2(200)
	// saveData(data)
	data := loadData()
	NewPointChart(data)

	// n.Calculate(data.Inputs)

	lossData := make([]float64, 0)
	func() {
		defer func() {
			if err := recover(); err != nil {
				fmt.Println("err -> ", err)
			}
		}()
		lossData = n.Train(data.Inputs, data.Expects, 10000, 0.01, 0.01)

		// outs := n.Calculate(data.Inputs)
		// for i, item := range outs {
		// 	loss := cnn.LogisticDiff.Loss(item, data.Expects[i])
		// 	lossData = append(lossData, loss)
		// }
	}()

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	cnn.GenerateLossChart(lossData, "test_out.png")
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

func NewPointChart(data *Data) {
	series := make([]chart.Series, 0)
	x1Value := make([]float64, 0)
	x2Value := make([]float64, 0)
	y1Value := make([]float64, 0)
	y2Value := make([]float64, 0)

	for i, _ := range data.Inputs {
		if data.Expects[i][0] == 1 {
			x1Value = append(x1Value, data.Inputs[i][0])
			y1Value = append(y1Value, data.Inputs[i][1])
		} else {
			x2Value = append(x2Value, data.Inputs[i][0])
			y2Value = append(y2Value, data.Inputs[i][1])
		}
	}

	series = append(series, chart.ContinuousSeries{
		Name: "red",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    3,
			DotColor:    chart.ColorRed,
		},
		XValues: x1Value,
		YValues: y1Value,
	})
	series = append(series, chart.ContinuousSeries{
		Name: "blue",
		Style: chart.Style{
			StrokeWidth: chart.Disabled,
			DotWidth:    3,
			DotColor:    chart.ColorBlue,
		},
		XValues: x2Value,
		YValues: y2Value,
	})

	// series = append(series, chart.ContinuousSeries{
	// 	Style: chart.Style{
	// 		StrokeColor: chart.GetDefaultColor(1),
	// 	},
	// 	XValues: []float64{0, 0.008886, 1},
	// 	YValues: []float64{-0.009462, 0, 1.055400},
	// })
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

func generateData(size int) *Data {
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	for {
		x, y := randPoint(1)
		if x-y < 0.05 && x-y > -0.05 {
			continue
		}
		inputs = append(inputs, []float64{x, y})
		if y > x {
			expects = append(expects, []float64{1})
		} else {
			expects = append(expects, []float64{0})
		}
		if len(inputs) == size {
			break
		}
	}

	return &Data{
		Inputs:  inputs,
		Expects: expects,
	}
}

func generateDataV2(size int) *Data {
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	b := false
	for {
		var x, y float64
		if b {
			x, y = randPointV2(1, 2, 2.8)
			expects = append(expects, []float64{1})
		} else {
			x, y = randPointV2(1, 2.8, 2)
			expects = append(expects, []float64{0})
		}

		inputs = append(inputs, []float64{x, y})
		b = !b

		if len(inputs) == size {
			break
		}
	}

	return &Data{
		Inputs:  inputs,
		Expects: expects,
	}
}

func saveData(data *Data) {
	b, _ := json.Marshal(data)
	f, err := os.Create("data.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	f.Write(b)
}

func loadData() *Data {
	f, err := os.Open("data.json")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	b, _ := ioutil.ReadAll(f)
	var data Data
	if err := json.Unmarshal(b, &data); err != nil {
		panic(err)
	}
	return &data
}

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func randPointV2(m float64, x, y float64) (float64, float64) {
	return round2((rand.Float64()-0.5)*m + x), round2((rand.Float64()-0.5)*m + y)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 0)
	return v
}
