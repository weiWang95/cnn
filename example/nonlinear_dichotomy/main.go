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

func getNY(x float64, n, i int, wm cnn.WeightMap) float64 {
	return getY(x, wm[n][i][fmt.Sprintf("%d-0", n-1)], wm[n][i][fmt.Sprintf("%d-1", n-1)], wm[n][i]["weight"])
}

func getY(x, w1, w2, b float64) float64 {
	return (0.5 - b - w1) * x / w2
}

func main() {
	n := cnn.NewNeuralNetwork([]int64{2, 4, 1}, []cnn.IActive{cnn.ReLU, cnn.Sigmoid}, cnn.LogisticDiff)

	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-9.339675985289869,"0-1":-0.43687579256207354,"weight":9.045699258569176},{"0-0":6.357655467537005,"0-1":11.81587087910318,"weight":-11.152105634279472},{"0-0":-0.552108729263139,"0-1":-0.00848789196856887,"weight":-5.098758458912421},{"0-0":19.305465948000837,"0-1":14.892532110387135,"weight":-16.805092658565364}],[{"1-0":11.884095918583533,"1-1":10.17399735120314,"1-2":8.937106062174276,"1-3":7.3570186901821275,"weight":-18.975022484922675}]]`
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-7.857271245302127,"0-1":1.3517645971053942,"weight":4.9976568581148015},{"0-0":1.0567078783952124,"0-1":6.640258827123178,"weight":-6.2224483055121595},{"0-0":-0.5385288277263268,"0-1":0.0072007912736191705,"weight":-4.9556206104821445},{"0-0":20.652923684432793,"0-1":16.242423284641383,"weight":-17.379444545492404}],[{"1-0":17.652140869184322,"1-1":15.504545875618316,"1-2":14.242732199669302,"1-3":12.27103578970768,"weight":-27.956968601120593}]]`
	// str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":-7.852474782041057,"0-1":1.8679142417395187,"weight":4.566121755599466},{"0-0":0.5527103869069345,"0-1":6.357856553809318,"weight":-5.889422867303683},{"0-0":-0.5698105029516644,"0-1":-0.03155083276321978,"weight":-5.257310560196139},{"0-0":21.990447671455765,"0-1":17.655802464556963,"weight":-18.729819962196423}],[{"1-0":22.016278392228863,"1-1":20.588702512275344,"1-2":19.039765097701366,"1-3":17.186677982960223,"weight":-36.12728041024525}]]`
	str := `[[{"0-0":1,"weight":0},{"0-0":1,"weight":0}],[{"0-0":0.4503800168668082,"0-1":0.8324029426048702,"weight":-0.29352263681338875},{"0-0":0.7774564389344208,"0-1":0.5403189962741372,"weight":-0.28668049047531197},{"0-0":0.44469667582021155,"0-1":0.8673363845857504,"weight":-0.2829007533716387},{"0-0":0.3951699179896143,"0-1":0.9012570130293864,"weight":-0.2825510997278432}],[{"1-0":1.3867990055649868,"1-1":1.2807779880721657,"1-2":1.366839335158994,"1-3":1.3532022757355595,"weight":-3.862908690749844}]]`
	var wm cnn.WeightMap
	json.Unmarshal([]byte(str), &wm)
	n.ApplyWeight(wm)

	// data := generateData(500)
	// saveData(data)
	data := loadData()
	NewPointChart(data, wm)

	lossData := n.Train(data.Inputs, data.Expects, 200, 0.005, 0.01)

	ws := n.ExportWeight()
	b, _ := json.Marshal(ws)
	fmt.Println("ws -> ", string(b))

	cnn.GenerateLossChart(lossData, "test_out.png")
}

func randPoint(m float64) (float64, float64) {
	return round2(rand.Float64() * m), round2(rand.Float64() * m)
}

func round2(v float64) float64 {
	v, _ = strconv.ParseFloat(fmt.Sprintf("%.02f", v), 0)
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
		if y > x && r-x < y {
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

func NewPointChart(data *Data, wm cnn.WeightMap) {
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
