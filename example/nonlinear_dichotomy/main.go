package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/wcharczuk/go-chart/v2"
	"github.com/weiWang95/cnn"
)

func getNY(x float64, n, i int, wm cnn.WeightMap) float64 {
	return getY(x, wm[n][i][fmt.Sprintf("%d-0", n-1)], wm[n][i][fmt.Sprintf("%d-1", n-1)], wm[n][i]["weight"])
}

func getY(x, w1, w2, b float64) float64 {
	return (0.5 - b - w1) * x / w2
}

func main() {
	rand.Seed(time.Now().UnixNano())
	n := cnn.NewNeuralNetwork([]int64{2, 4, 1}, []cnn.IActive{cnn.LeakyReLU, cnn.Sigmoid}, cnn.LogisticDiff)

	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-0.42623881529928515,"0-1":-1.2662420320501753,"weight":-0.4618140695180213},{"0-0":-0.813579366965817,"0-1":-1.0647499327061312,"weight":-0.3557059077141387},{"0-0":-1.7325139832205683,"0-1":-3.558990999482263,"weight":-0.41523105689009165},{"0-0":-1.3902028577164252,"0-1":-1.1454229080151248,"weight":-0.12889147678091667}],[{"1-0":0.21132839643936813,"1-1":0.4724259531971717,"1-2":0.8839977210919784,"1-3":0.42164757853677637,"weight":0.036590179686797744},{"1-0":0.5163542349405961,"1-1":0.3458006607576927,"1-2":0.8577581342870433,"1-3":-0.02229141035306279,"weight":-0.29254329541182755},{"1-0":0.9049583210477231,"1-1":0.27388502559495603,"1-2":0.38274888576839294,"1-3":-0.04827892714110561,"weight":-0.44293757063884975},{"1-0":0.15871749882724664,"1-1":0.18612022241400436,"1-2":0.15558884041171722,"1-3":-0.10351322327833107,"weight":-0.5991512294335709}],[{"2-0":-0.3584482425692751,"2-1":0.03093842328081094,"2-2":-0.02948563844536686,"2-3":0.3894405555847553,"weight":-0.8208762429056098}]]`
	// var wm cnn.WeightMap
	// json.Unmarshal([]byte(str), &wm)
	// n.ApplyWeight(wm)

	// data := generateData(2000)
	// saveData(data)
	data := loadData()
	NewPointChart(data)

	// lr := cnn.NewStepLR(0.1, 0.1, 0.001, 20)
	lr := cnn.NewConstLR(0.03)
	// lossData := n.Train(data.Inputs, data.Expects, 10, 32, lr, 0.1)
	lossData, testLossData := n.TrainWithTest(data.Inputs[0:1000], data.Expects[0:1000], data.Inputs[1000:], data.Expects[1000:], 1000, 32, lr, 0.1)

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	// cnn.GenerateLossChart(lossData, "test_out.png")

	fmt.Println(len(lossData), len(testLossData))
	cnn.GenerateAllLossChart(lossData, testLossData, "test_out.png")

	// var count int
	// for i, input := range data.Inputs {
	// 	outs := n.Compute(input...)
	// 	if (outs[0] > 0.5 && data.Expects[i][0] > 0.5) || (outs[0] < 0.5 && data.Expects[i][0] < 0.5) {
	// 		count += 1
	// 	}
	// }
	// fmt.Println(float64(count) / float64(len(data.Inputs)))
}

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 10)
	return v
}

type Data struct {
	Inputs  [][]float64 `json:"inputs"`
	Expects [][]float64 `json:"expects"`
}

func generateData(size int) *Data {
	r := 1.0
	inputs := make([][]float64, 0)
	expects := make([][]float64, 0)
	for {
		x, y := randPoint(r)
		if (x-y < r*0.05 && x-y > -r*0.05) || (1-x-y < r*0.05 && 1-x-y > -r*0.05) {
			continue
		}
		inputs = append(inputs, []float64{x, y})
		if (y > x && r-x < y) || (y < x && r-x > y) {
			expects = append(expects, []float64{0.99})
		} else {
			expects = append(expects, []float64{0.01})
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

func NewPointChart(data *Data) {
	series := make([]chart.Series, 0)
	x1Value := make([]float64, 0)
	x2Value := make([]float64, 0)
	y1Value := make([]float64, 0)
	y2Value := make([]float64, 0)

	for i, _ := range data.Inputs {
		if data.Expects[i][0] > 0.5 {
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

	// for i, _ := range wm[1] {
	// 	series = append(series, chart.ContinuousSeries{
	// 		Style: chart.Style{
	// 			StrokeColor: chart.GetDefaultColor(1),
	// 		},
	// 		XValues: []float64{0, 1},
	// 		YValues: []float64{getNY(0, 1, i, wm), getNY(1, 1, i, wm)},
	// 	})
	// }
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
