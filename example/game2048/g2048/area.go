package g2048

import (
	"fmt"
	"math/rand"
)

type Area struct {
	w    int
	h    int
	seed int64

	data    [][]uint
	newData [][]uint
	score   int
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

	a.newData = make([][]uint, a.h)
	for i, _ := range a.newData {
		a.newData[i] = make([]uint, a.w)
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

func (a *Area) isFull(data [][]uint) bool {
	for _, item := range data {
		for _, i := range item {
			if i == 0 {
				return false
			}
		}
	}

	return true
}

func (a *Area) left() {
	a.copyData(a.data, a.newData)
	a.newData = a.doLeft(a.newData)
}

func (a *Area) doLeft(data [][]uint) [][]uint {
	for y, _ := range data {
		position := 0

		for x := 1; x < a.w; x++ {
			if data[y][x] == 0 {
				continue
			}

			if data[y][position] == 0 {
				data[y][position] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[y][position] == data[y][x] {
				data[y][position] = data[y][position] * 2
				data[y][x] = 0
				position++
			} else {
				position++
				if position != x {
					data[y][position] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data
}

func (a *Area) right() {
	a.copyData(a.data, a.newData)
	a.newData = a.doRight(a.newData)
}

func (a *Area) doRight(data [][]uint) [][]uint {
	for y, _ := range data {
		position := a.w - 1

		for x := position - 1; x >= 0; x-- {
			if data[y][x] == 0 {
				continue
			}

			if data[y][position] == 0 {
				data[y][position] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[y][position] == data[y][x] {
				data[y][position] = data[y][position] * 2
				data[y][x] = 0
				position--
			} else {
				position--
				if position != x {
					data[y][position] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data
}

func (a *Area) up() {
	a.copyData(a.data, a.newData)
	a.newData = a.doUp(a.newData)
}

func (a *Area) doUp(data [][]uint) [][]uint {
	for x := 0; x < a.w; x++ {
		position := 0
		for y := position + 1; y < a.h; y++ {
			if data[y][x] == 0 {
				continue
			}

			if data[position][x] == 0 {
				data[position][x] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[position][x] == data[y][x] {
				data[position][x] = data[position][x] * 2
				data[y][x] = 0
				position++
			} else {
				position++
				if position != y {
					data[position][x] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data
}

func (a *Area) down() {
	a.copyData(a.data, a.newData)
	a.newData = a.doDown(a.newData)
}

func (a *Area) doDown(data [][]uint) [][]uint {
	for x := 0; x < a.w; x++ {
		position := a.h - 1
		for y := position - 1; y >= 0; y-- {
			if data[y][x] == 0 {
				continue
			}

			if data[position][x] == 0 {
				data[position][x] = data[y][x]
				data[y][x] = 0
				continue
			}

			if data[position][x] == data[y][x] {
				data[position][x] = data[position][x] * 2
				data[y][x] = 0
				position--
			} else {
				position--
				if position != y {
					data[position][x] = data[y][x]
					data[y][x] = 0
				}
			}
		}
	}

	return data
}

func (a *Area) apply() {
	a.copyData(a.newData, a.data)
}

func (a *Area) copyData(source, target [][]uint) {
	for i, _ := range source {
		for j, _ := range source[i] {
			target[i][j] = source[i][j]
		}
	}
}

func (a *Area) computeScore(data [][]uint) int {
	var sum int
	for _, row := range data {
		for _, item := range row {
			sum += int(item)
		}
	}
	return sum
}

func (a *Area) SetScore(sum int) {
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
