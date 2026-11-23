// process-status reads the Server's kernel high-water mark in its own container
// PID namespace, including when the selected Podman engine runs in a VM.
package main

import (
	"fmt"
	"os"
	"strings"
)

func main() {
	status, err := os.ReadFile("/proc/1/status")
	if err != nil {
		fmt.Fprintln(os.Stderr, "cannot read container process status")
		os.Exit(1)
	}
	for _, line := range strings.Split(string(status), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			fmt.Println(line)
			return
		}
	}
	fmt.Fprintln(os.Stderr, "kernel did not report the Server high-water mark")
	os.Exit(1)
}
