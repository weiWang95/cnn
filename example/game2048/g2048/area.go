package g2048

import (
	"fmt"
	"math/rand"
)

type Area struct {
	w    int
	h    int
	seed int64

	data  [][]uint
	score int
}

func newArea(w, h int, seed int64) *Area {
	return &Area{
		w:    w,
		h:    h,
		seed: seed,
	}
}

func (a *Area) init() {
	rand.Seed(a.seed)

	a.data = make([][]uint, a.h)
	for i, _ := range a.data {
		a.data[i] = make([]uint, a.w)
	}
	a.score = 0
}

func (a *Area) random() {
	v := 2
	if rand.Intn(2) == 1 {
		v = 4
	}

	for {
		x := rand.Intn(a.w)
		y := rand.Intn(a.h)
		if a.data[y][x] == 0 {
			a.data[y][x] = uint(v)
			break
		}
	}
}

func (a *Area) isFull() bool {
	for _, item := range a.data {
		for _, i := range item {
			if i == 0 {
				return false
			}
		}
	}

	return true
}

func (a *Area) left() {
	for y, _ := range a.data {
		position := 0

		for x := 1; x < a.w; x++ {
			if a.data[y][x] == 0 {
				continue
			}

			if a.data[y][position] == 0 {
				a.data[y][position] = a.data[y][x]
				a.data[y][x] = 0
				continue
			}

			if a.data[y][position] == a.data[y][x] {
				a.data[y][position] = a.data[y][position] * 2
				a.data[y][x] = 0
				position++
			} else {
				position++
				if position != x {
					a.data[y][position] = a.data[y][x]
					a.data[y][x] = 0
				}
			}
		}
	}
}

func (a *Area) right() {
	for y, _ := range a.data {
		position := a.w - 1

		for x := position - 1; x >= 0; x-- {
			if a.data[y][x] == 0 {
				continue
			}

			if a.data[y][position] == 0 {
				a.data[y][position] = a.data[y][x]
				a.data[y][x] = 0
				continue
			}

			if a.data[y][position] == a.data[y][x] {
				a.data[y][position] = a.data[y][position] * 2
				a.data[y][x] = 0
				position--
			} else {
				position--
				if position != x {
					a.data[y][position] = a.data[y][x]
					a.data[y][x] = 0
				}
			}
		}
	}
}

func (a *Area) up() {
	for x := 0; x < a.w; x++ {
		position := 0
		for y := position + 1; y < a.h; y++ {
			if a.data[y][x] == 0 {
				continue
			}

			if a.data[position][x] == 0 {
				a.data[position][x] = a.data[y][x]
				a.data[y][x] = 0
				continue
			}

			if a.data[position][x] == a.data[y][x] {
				a.data[position][x] = a.data[position][x] * 2
				a.data[y][x] = 0
				position++
			} else {
				position++
				if position != y {
					a.data[position][x] = a.data[y][x]
					a.data[y][x] = 0
				}
			}
		}
	}
}

func (a *Area) down() {
	for x := 0; x < a.w; x++ {
		position := a.h - 1
		for y := position - 1; y >= 0; y-- {
			if a.data[y][x] == 0 {
				continue
			}

			if a.data[position][x] == 0 {
				a.data[position][x] = a.data[y][x]
				a.data[y][x] = 0
				continue
			}

			if a.data[position][x] == a.data[y][x] {
				a.data[position][x] = a.data[position][x] * 2
				a.data[y][x] = 0
				position--
			} else {
				position--
				if position != y {
					a.data[position][x] = a.data[y][x]
					a.data[y][x] = 0
				}
			}
		}
	}
}

func (a *Area) computeScore() {
	var sum int
	for _, row := range a.data {
		for _, item := range row {
			sum += int(item)
		}
	}
	a.score = sum
}

func (a *Area) Score() int {
	return a.score
}

func (a *Area) inspect() {

	for y := 0; y < a.h; y++ {
		for x := 0; x < a.w; x++ {
			fmt.Printf("% 4d ", a.data[y][x])
		}
		fmt.Print("\n\n")
	}
	fmt.Print("\n")
}
