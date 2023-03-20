package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/weiWang95/cnn"
	"github.com/weiWang95/cnn/example/game2048/g2048"
)

const w = 3

var (
	freq             = 5
	batchSize        = 128
	gamma            = 1.0
	epsilon          = 0.9
	incEpsilon       = 0.0001
	rate             = 0.001
	maxEx      int64 = 6000
	minEx            = 500
)

var operate = []g2048.Direction{g2048.DirectionUp, g2048.DirectionDown, g2048.DirectionLeft, g2048.DirectionRight}

func main() {
	// logrus.SetLevel(logrus.TraceLevel)
	logrus.SetLevel(logrus.DebugLevel)
	// Train()

	rand.Seed(time.Now().UnixNano())
	// for i := 0; i < 5; i++ {
	run(false)
	// }
}

func Train() {
	ctx, cancel := context.WithCancel(context.Background())
	done := train(ctx)
	// train2()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGQUIT, syscall.SIGTERM, syscall.SIGINT)
	select {
	case <-sig:
		logrus.Info("receive sigterm")
		cancel()
		time.Sleep(3 * time.Second)
		logrus.Info("over 3 sec")
	case <-done:
	}

	logrus.Info("start test run")

	for i := 0; i < 5; i++ {
		logrus.Infof("start run: %d", i)
		run(false)
	}
}

func run(print bool) {
	q := cnn.NewNeuralNetwork([]int64{w * w, 128, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff, cnn.WithSoftmax())
	// loadModel(q, "model_3x3.json")
	loadModel(q, "model.json")

	env := g2048.NewGame(w, w, time.Now().UnixNano())
	env.Init()
	if print {
		env.Inspect()
	}

	step := 0
	m := make(map[g2048.Direction]int)
	arr := make([]g2048.Direction, 0)
	for {
		step += 1

		d := predict(q, normalizeInput(env.State()))
		cont, reward := env.Operate(d)
		// ex := operateEnv(env, d)
		m[d] = m[d] + 1
		arr = append(arr, d)

		if print {
			env.Inspect()
		}

		// 结束 或者 无效移动
		if !cont || reward == g2048.NoMoveReward {
			break
		}
	}

	logrus.Debugf("step:%d score:%d max:%d, %v %v\n", step, env.Score(), env.Max(), m, arr)
}

func train(ctx context.Context) <-chan struct{} {
	var stop bool
	done := make(chan struct{})

	go func() {
		rand.Seed(time.Now().UnixNano())
		result := make([]float64, 0)
		q := cnn.NewNeuralNetwork([]int64{w * w, 128, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff, cnn.WithDefaultInputWeight(0.0001), cnn.WithDefaultWeight(0.0001))
		// loadModel(q, "model.json")
		o1 := q.ExportWeight()

		q2 := cnn.NewNeuralNetwork([]int64{w * w, 128, 4}, []cnn.IActive{cnn.ReLU, cnn.ReLU, cnn.ReLU}, cnn.SquareDiff)
		q2.ApplyWeight(o1)

		pool := NewExPool(maxEx)
		seed := time.Now().UnixNano()
		env := g2048.NewGame(w, w, seed)
		env.Init()

		defer func() {
			logrus.Infof("stoped! %d", len(result))
			if err := saveModel(q2); err != nil {
				logrus.Trace("err: ", err)
			}

			cnn.GenerateLossChart(result, "loss_out.jpg")

			close(done)
		}()

		logrus.Debug("init pool")
		for {
			if pool.Len() >= minEx {
				break
			}

			ex := operateEnv(env, randOperate())
			pool.Push(ex)

			if ex.End {
				env.Init()
				// env.SetSeed(seed)
			}
		}

		logrus.Debug("start run")
		for i := 0; i < 2000; i++ {
			if stop {
				break
			}
			logrus.Debugf("start run: %d eps:%.06f \n", i, epsilon)
			env.Init()
			// env.SetSeed(seed)
			step := 0
			reward := 0.0
			actionMap := make(map[g2048.Direction]int)
			actionArr := make([]g2048.Direction, 0)

			for {
				step += 1
				d1 := sampleOperate(q2, env.State())
				actionMap[d1] = actionMap[d1] + 1
				actionArr = append(actionArr, d1)

				ex := operateEnv(env, d1)
				pool.Push(ex)
				reward += ex.Reward

				if pool.Len() > minEx && step%freq == 0 {
					avgLoss := 0.0

					for _, item := range pool.Sample(batchSize) {
						// qouts := q.Compute(normalizeInput(item.State)...)
						// idx := cnn.ArgMax(qouts...)

						// nextState, score, _ := env.TryOperate(item.State, d)
						// reward := float64(score - env.Score())
						reward := item.Reward * 0.01
						if !item.End && item.Reward != g2048.NoMoveReward {
							inputs := normalizeInput(item.NextState)

							qouts := q.Compute(inputs...)
							idx := cnn.ArgMax(qouts...)

							logrus.Tracef("inputs: %v\n", inputs)
							q2outs := q2.Compute(inputs...)
							logrus.Tracef("outs: %v idx:%d\n", q2outs, idx)

							logrus.Tracef("%.06f + %.06f * %.06f", reward, gamma, q2outs[idx])
							reward += gamma * q2outs[idx]
							logrus.Tracef(" => %.06f\n", reward)
						}

						qouts := q.Compute(normalizeInput(item.State)...)
						loss := cnn.SquareDiff.Loss([]float64{qouts[item.Action]}, []float64{reward})
						logrus.Tracef("loss: %.06f [%.06f %.06f]\n", loss, qouts[item.Action], reward)
						avgLoss += loss

						qouts[item.Action] = reward
						q.BP(rate, qouts...)
						// expects := []float64{0, 0, 0, 0}
						// expects[item.Action] = reward
						// q.BP(rate, expects...)
					}

					avgLoss = avgLoss / float64(batchSize)
					logrus.Tracef("avg loss -> %.06f\n", avgLoss/float64(batchSize))
					if avgLoss < 1 {
						result = append(result, avgLoss)
					}

					syncWeight(q, q2)
				}

				if ex.End {
					logrus.Debugf("step:%v score:%v max:%v reward: %v action:%v flow:%v\n", step, env.Score(), env.Max(), reward, actionMap, actionArr)
					break
				} else if ex.Reward == g2048.NoMoveReward {
					for _, item := range []int{0, 1, 2, 3} {
						ex := operateEnv(env, g2048.Direction(item))
						pool.Push(ex)
						if ex.Reward != g2048.NoMoveReward {
							reward += ex.Reward
							break
						}
					}
				}
			}
		}
	}()

	go func() {
		select {
		case <-done:
			logrus.Info("done")
		case <-ctx.Done():
			logrus.Info("stoping")
			stop = true
		}
	}()

	return done
}

func predict(n *cnn.NeuralNetwork, state []float64) g2048.Direction {
	// logrus.Tracef("predict input: %v", state)
	outs := n.Compute(state...)
	d := cnn.ArgMax(outs...)
	logrus.Tracef("predict: %v -> %d", outs, d)
	return g2048.Direction(d)
}

func sampleOperate(n *cnn.NeuralNetwork, state []uint) g2048.Direction {
	epsilon = math.Max(0.01, epsilon-incEpsilon)

	r := rand.Float64()
	if r < epsilon {
		return randOperate()
	} else {
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

func printOuts(outs []float64) []string {
	strs := make([]string, 0, len(outs))
	for _, item := range outs {
		strs = append(strs, fmt.Sprintf("%.06f", item))
	}

	return strs
}

func operateEnv(env *g2048.Game, d g2048.Direction) Ex {
	ex := Ex{State: env.State(), Action: uint8(d)}
	coun, score := env.Operate(d)
	ex.End = !coun
	ex.Reward = float64(score)
	ex.NextState = env.State()

	return ex
}

func tryOperateEnv(env *g2048.Game, d g2048.Direction) Ex {
	ex := Ex{State: env.State(), Action: uint8(d)}
	nextState, _, stepScore, isEnd := env.TryOperate(ex.State, d)
	ex.End = isEnd
	ex.Reward = float64(stepScore)
	ex.NextState = nextState

	return ex
}

func normalize(d int) float64 {
	if d == 0 {
		return 0
	}

	return math.Log(float64(d))
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
	// logrus.Trace(string(bs))
	return ioutil.WriteFile("model.json", bs, os.ModePerm)
}

func loadModel(n *cnn.NeuralNetwork, name string) error {
	bs, err := ioutil.ReadFile(name)
	if err != nil {
		return err
	}

	var m cnn.WeightMap
	if err := json.Unmarshal(bs, &m); err != nil {
		return err
	}

	n.ApplyWeight(m)
	return nil
}
