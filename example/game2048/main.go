package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/weiWang95/cnn"
	"github.com/weiWang95/cnn/example/game2048/g2048"
)

const w = 4

var (
	freq       = 5
	batchSize  = 16
	gamma      = 0.99
	epsilon    = 0.9
	incEpsilon = 0.0001
	rate       = 0.01
	minEx      = 500
)

var operate = []g2048.Direction{g2048.DirectionUp, g2048.DirectionDown, g2048.DirectionLeft, g2048.DirectionRight}

func main() {
	// run()
	train()
}

func run() {
	q := cnn.NewNeuralNetwork([]int64{16, 32, 32, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff, cnn.WithSoftmax())
	loadModel(q)

	env := g2048.NewGame(w, w, time.Now().UnixNano())
	env.Init()

	for {
		env.Inspect()

		d := predict(q, normalizeInput(env.State()))
		ex := operateEnv(env, d)

		if ex.End {
			break
		}

		time.Sleep(time.Second)
	}
}

func train() {
	q := cnn.NewNeuralNetwork([]int64{16, 64, 64, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff, cnn.WithSoftmax())
	o1 := q.ExportWeight()

	q2 := cnn.NewNeuralNetwork([]int64{16, 64, 64, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff, cnn.WithSoftmax())
	q2.ApplyWeight(o1)

	pool := NewExPool(10000)
	env := g2048.NewGame(w, w, time.Now().UnixNano())
	env.Init()

	fmt.Println("init pool")
	for {
		if pool.Len() >= minEx {
			break
		}

		ex := operateEnv(env, randOperate())
		pool.Push(ex)

		if ex.End {
			env.Init()
			env.SetSeed(time.Now().UnixNano())
		}
	}

	fmt.Println("start run")
	for i := 0; i < 200; i++ {
		fmt.Println("start run: ", i)
		env.Init()
		env.SetSeed(time.Now().UnixNano())
		step := 0

		for {
			step += 1
			ex := operateEnv(env, sampleOperate(q2, pool, env.State()))
			pool.Push(ex)

			if pool.Len() > minEx && step%freq == 0 {
				for _, item := range pool.Sample(batchSize) {
					learn(q, item)
				}
			}

			if ex.End {
				fmt.Printf("step:%v score:%v\n", step, env.Score())
				break
			}
		}

		syncWeight(q, q2)
	}

	// os1, _ := json.Marshal(q.ExportWeight())
	// fmt.Println(string(os1))

	// os2, _ := json.Marshal(q2.ExportWeight())
	// fmt.Println(string(os2))
	saveModel(q2)
}

func predict(n *cnn.NeuralNetwork, state []float64) g2048.Direction {
	out := n.Compute(state...)
	return getOperate(out)
}

func learn(n *cnn.NeuralNetwork, ex Ex) {
	reward := ex.Reward
	if !ex.End {
		reward = ex.Reward + gamma*n.Compute(normalizeInput(ex.State)...)[0]
	}

	expect := []float64{0.01, 0.01, 0.01, 0.01}
	expect[ex.Action] = 0.99
	n.BP(rate*reward, expect...)
}

func sampleOperate(n *cnn.NeuralNetwork, pool *ExPool, state []uint) g2048.Direction {
	r := rand.Float64()
	if r < epsilon {
		return randOperate()
	} else {
		epsilon = math.Max(0.01, epsilon-incEpsilon)
		return predict(n, normalizeInput(state))
	}
}

func randOperate() g2048.Direction {
	return operate[rand.Intn(len(operate))]
}

func syncWeight(n *cnn.NeuralNetwork, target *cnn.NeuralNetwork) {
	w := n.ExportWeight()
	target.ApplyWeight(w)
}

func getInputs(g *g2048.Game) []float64 {
	inputs := make([]float64, 0, w*w)
	for i := 0; i < w; i++ {
		for j := 0; j < w; j++ {
			inputs = append(inputs, float64(g.Data(i, j)))
		}
	}
	return inputs
}

func getOperate(out []float64) g2048.Direction {
	var maxIdx int
	var max float64
	for i, item := range out {
		if item > max {
			maxIdx = i
			max = item
		}
	}

	return operate[maxIdx]
}

func printOuts(outs []float64) []string {
	strs := make([]string, 0, len(outs))
	for _, item := range outs {
		strs = append(strs, fmt.Sprintf("%.06f", item))
	}

	return strs
}

func operateEnv(env *g2048.Game, d g2048.Direction) Ex {
	ex := Ex{State: env.State(), Action: uint8(d)}
	oldReward := normalize(env.Score())
	ex.End = !env.Operate(d)
	ex.Reward = normalize(env.Score()) - oldReward
	ex.NextState = env.State()

	return ex
}

func normalize(d int) float64 {
	v := math.Log(float64(d+1)) / 16
	if v == 0 {
		return 0.000001
	}

	return v
}

func normalizeInput(inputs []uint) []float64 {
	o := make([]float64, 0, len(inputs))
	for _, item := range inputs {
		o = append(o, normalize(int(item)))
	}
	return o
}

func saveModel(n *cnn.NeuralNetwork) error {
	d := n.ExportWeight()
	bs, _ := json.Marshal(d)
	return ioutil.WriteFile("model.json", bs, os.ModePerm)
}

func loadModel(n *cnn.NeuralNetwork) error {
	bs, err := ioutil.ReadFile("model.json")
	if err != nil {
		return err
	}

	var m cnn.WeightMap
	json.Unmarshal(bs, &m)

	n.ApplyWeight(m)
	return nil
}
