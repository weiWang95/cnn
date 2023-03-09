package g2048

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
)

type Direction int

const (
	DirectionUp = iota
	DirectionDown
	DirectionLeft
	DirectionRight
)

type Game struct {
	area *Area

	seed int64
	done chan int
}

func NewGame(w, h int, seed int64) *Game {
	return &Game{
		seed: seed,
		area: newArea(w, h, seed),
		done: make(chan int),
	}
}

func (g *Game) Start() {
	g.info()
	g.Init()

	sigterm := make(chan os.Signal, 1)
	signal.Notify(sigterm, syscall.SIGQUIT, syscall.SIGTERM, syscall.SIGINT)

	go g.inputListen()
	select {
	case <-sigterm:
	case <-g.done:
	}
}

func (g *Game) Init() {
	g.area.init()
	g.area.random()
}

func (g *Game) SetSeed(seed int64) {
	g.seed = seed
}

func (g *Game) Over() {
	close(g.done)
}

func (g *Game) Operate(d Direction) bool {
	switch d {
	case DirectionUp:
		g.area.up()
	case DirectionDown:
		g.area.down()
	case DirectionLeft:
		g.area.left()
	case DirectionRight:
		g.area.right()
	default:
		return true
	}

	g.area.computeScore()

	if g.area.isFull() {
		return false
	}

	g.area.random()

	return true
}

func (g *Game) Data(x, y int) uint {
	return g.area.data[y][x]
}

func (g *Game) State() []uint {
	state := make([]uint, 0, g.area.w*g.area.h)
	for y := 0; y < g.area.h; y++ {
		for x := 0; x < g.area.w; x++ {
			state = append(state, g.area.data[y][x])
		}
	}
	return state
}

func (g *Game) SetData(data [][]uint) {
	for y := 0; y < len(g.area.data); y++ {
		for x := 0; x < len(g.area.data[y]); x++ {
			g.area.data[y][x] = data[y][x]
		}
	}
}

func (g *Game) Score() int {
	return g.area.Score()
}

func (g *Game) info() {
	fmt.Printf("2048 Game\n")
	fmt.Printf("w a s d 移动\n")
}

func (g *Game) Inspect() {
	g.area.inspect()
}

func (g *Game) inputListen() {
	var operate string
	for {
		g.Inspect()

		_, err := fmt.Scanln(&operate)
		if err != nil {
			return
		}

		switch operate {
		case "w":
			g.area.up()
		case "s":
			g.area.down()
		case "a":
			g.area.left()
		case "d":
			g.area.right()
		case "quit":
			g.Over()
			return
		default:
			continue
		}

		g.area.computeScore()

		if g.area.isFull() {
			g.Over()
			return
		}

		g.area.random()
	}
}
