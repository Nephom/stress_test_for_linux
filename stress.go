package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Configuration structure
type Config struct {
	Debug bool `json:"debug"`
}

// Test result structure
type TestResult struct {
	CPU    string
	DIMM   string
	HDD    string
}

// DiskTestConfig saves disk test configuration
type DiskTestConfig struct {
	MountPoints []string // Mount points to test
	FileSize    int64    // Test file size (bytes)
	TestMode    string   // "sequential" or "random"
	BlockSize   int64    // Block size for read/write operations
}

type CPUPerformance struct {
	GFLOPS float64
}

type MemoryPerformance struct {
	ReadSpeed  float64 // in MB/s
	WriteSpeed float64 // in MB/s
}

type DiskPerformance struct {
	ReadSpeed  float64 // in MB/s
	WriteSpeed float64 // in MB/s
	MountPoint string
	Mode       string
	BlockSize  int64
}

type PerformanceStats struct {
	CPU    CPUPerformance
	Memory MemoryPerformance
	Disk   []DiskPerformance
	mu     sync.Mutex
}

// NUMAInfo holds information about NUMA nodes
type NUMAInfo struct {
	NumNodes int
	NodeCPUs [][]int
}

// Get NUMA node information (Linux-specific)
func getNUMAInfo() (NUMAInfo, error) {
	info := NUMAInfo{
		NumNodes: 1, // Default to 1 node if detection fails
		NodeCPUs: make([][]int, 0),
	}

	// Check if /sys/devices/system/node exists
	nodeDir := "/sys/devices/system/node"
	if _, err := os.Stat(nodeDir); os.IsNotExist(err) {
		// NUMA information not available, create default mapping
		cpus := make([]int, runtime.NumCPU())
		for i := 0; i < runtime.NumCPU(); i++ {
			cpus[i] = i
		}
		info.NodeCPUs = append(info.NodeCPUs, cpus)
		return info, nil
	}

	// Read node directories
	files, err := os.ReadDir(nodeDir)
	if err != nil {
		return info, err
	}

	for _, file := range files {
		if !file.IsDir() || !strings.HasPrefix(file.Name(), "node") {
			continue
		}

		// Parse node ID
		nodeID, err := strconv.Atoi(strings.TrimPrefix(file.Name(), "node"))
		if err != nil {
			continue
		}

		// Ensure we have enough capacity in our slice
		if nodeID >= len(info.NodeCPUs) {
			newSize := nodeID + 1
			if len(info.NodeCPUs) < newSize {
				// Resize the slice
				oldSize := len(info.NodeCPUs)
				info.NodeCPUs = append(info.NodeCPUs, make([][]int, newSize-oldSize)...)
			}
		}

		// Read CPU list for this node
		cpuList, err := os.ReadFile(filepath.Join(nodeDir, file.Name(), "cpulist"))
		if err != nil {
			continue
		}

		// Parse CPU list (format can be like "0-3,7,9-11")
		cpus := make([]int, 0)
		for _, segment := range strings.Split(strings.TrimSpace(string(cpuList)), ",") {
			if strings.Contains(segment, "-") {
				// Range of CPUs
				parts := strings.Split(segment, "-")
				if len(parts) != 2 {
					continue
				}
				start, err := strconv.Atoi(parts[0])
				if err != nil {
					continue
				}
				end, err := strconv.Atoi(parts[1])
				if err != nil {
					continue
				}
				for i := start; i <= end; i++ {
					cpus = append(cpus, i)
				}
			} else {
				// Single CPU
				cpu, err := strconv.Atoi(segment)
				if err != nil {
					continue
				}
				cpus = append(cpus, cpu)
			}
		}

		info.NodeCPUs[nodeID] = cpus
	}

	// Set the actual number of NUMA nodes found
	info.NumNodes = 0
	for i := range info.NodeCPUs {
		if len(info.NodeCPUs[i]) > 0 {
			info.NumNodes = i + 1
		}
	}

	return info, nil
}

// Integer computation with NUMA awareness
func integerComputation(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, numaNode int, cpuID int, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()
	
	// Set CPU affinity if supported
	if runtime.GOOS == "linux" {
		// This is simplified - actual affinity setting would use cgo or syscalls
		// which is beyond the scope of this example
		if debug {
			logMessage(fmt.Sprintf("Running integer computation on NUMA node %d, CPU %d", numaNode, cpuID), debug)
		}
	}

	startTime := time.Now()
	operations := uint64(0)
	var x int64 = 1
	
	for {
		select {
		case <-stop:
			// Calculate GFLOPS
			duration := time.Since(startTime).Seconds()
			flops := float64(operations * 3) // 3 operations per iteration
			gflops := 0.0
			if duration > 0 { // 避免除以零
			    gflops = flops / duration / 1e9
		        }

			perfStats.mu.Lock()
			if gflops > perfStats.CPU.GFLOPS {
				perfStats.CPU.GFLOPS = gflops
			}
			perfStats.mu.Unlock()

			if debug {
				logMessage(fmt.Sprintf("Integer computation on CPU %d completed: %.2f GFLOPS", cpuID, gflops), debug)
			}
			return
		default:
			x += 3
			x *= 2
			x /= 2
			operations++
			
			if x < 0 {
				errorChan <- "Integer overflow detected!"
			}
		}
	}
}

func logError(errorChan chan string, testName, errorMsg string) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	fullError := fmt.Sprintf("[%s] [%s] ERROR: %s", timestamp, testName, errorMsg)
	
	// Log to file
	f, err := os.OpenFile("stress.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		defer f.Close()
		logger := log.New(f, "", 0)
		logger.Println(fullError)
	}
	
	// Also push to error channel
	errorChan <- fullError
}

// Float computation with NUMA awareness
func floatComputation(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, numaNode int, cpuID int, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()
	
	testName := "FloatComputation"
	
	// Set CPU affinity if supported
	if runtime.GOOS == "linux" {
		if debug {
			logMessage(fmt.Sprintf("Running float computation on NUMA node %d, CPU %d", numaNode, cpuID), debug)
		}
	}

	startTime := time.Now()
	operations := uint64(0)
	var x float64 = 1.1
	
	for {
		select {
		case <-stop:
			// Calculate GFLOPS
			duration := time.Since(startTime).Seconds()
			flops := float64(operations * 5) // 5 operations per iteration (sqrt*2, add, mult, div)
			gflops := 0.0
			if duration > 0 { // 避免除以零
			    gflops = flops / duration / 1e9
		        }

			perfStats.mu.Lock()
			if gflops > perfStats.CPU.GFLOPS {
				perfStats.CPU.GFLOPS = gflops
			}
			perfStats.mu.Unlock()

			if debug {
				logMessage(fmt.Sprintf("Float computation on CPU %d completed: %.2f GFLOPS", cpuID, gflops), debug)
			}
			return
		default:
			oldX := x
			x = math.Sqrt(x) * math.Sqrt(x)
			x += 1.00001
			x *= 1.00001
			x /= 1.00001
			operations++
			
			if math.IsNaN(x) {
				logError(errorChan, testName, fmt.Sprintf("NaN detected! Previous value: %.6f", oldX))
			}
		}
	}
}

// Vector computation with NUMA awareness
func vectorComputation(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, numaNode int, cpuID int, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()
	
	// Set CPU affinity if supported
	if runtime.GOOS == "linux" {
		if debug {
			logMessage(fmt.Sprintf("Running vector computation on NUMA node %d, CPU %d", numaNode, cpuID), debug)
		}
	}

	vecSize := 1024
	a := make([]float64, vecSize)
	b := make([]float64, vecSize)
	c := make([]float64, vecSize)
	
	for i := 0; i < vecSize; i++ {
		a[i] = float64(i)
		b[i] = float64(i) * 1.5
	}
	
	startTime := time.Now()
	iterations := uint64(0)
	
	for {
		select {
		case <-stop:
			// Calculate GFLOPS
			duration := time.Since(startTime).Seconds()
			flops := float64(iterations * uint64(vecSize) * 4) // 4 operations per vector element
			gflops := 0.0
			if duration > 0 { // 避免除以零
			    gflops = flops / duration / 1e9
		        }

			perfStats.mu.Lock()
			if gflops > perfStats.CPU.GFLOPS {
				perfStats.CPU.GFLOPS = gflops
			}
			perfStats.mu.Unlock()

			if debug {
				logMessage(fmt.Sprintf("Vector computation on CPU %d completed: %.2f GFLOPS", cpuID, gflops), debug)
			}
			return
		default:
			for i := 0; i < vecSize; i++ {
				c[i] = a[i] + b[i]*2.0 - math.Sqrt(b[i])
				if math.IsNaN(c[i]) {
					errorChan <- "Vector computation error detected!"
				}
			}
			iterations++
		}
	}
}

// Get system memory info
func getSystemMemory() (total, free uint64) {
	var info syscall.Sysinfo_t
	err := syscall.Sysinfo(&info)
	if err != nil {
		log.Fatalf("Failed to get system memory info: %v", err)
	}
	
	// In modern kernels, unit size is fixed at 1
	total = info.Totalram
	free = info.Freeram
	return
}

// Enhanced memory test with speed measurement
func memoryTest(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, memUsagePercent float64, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()
	
	totalMem, freeMem := getSystemMemory()
	targetMemBytes := uint64(float64(freeMem) * memUsagePercent)
	
	const arraySize = 10_000_000 // 10 million int64
	const bytesPerEntry = 8
	arraysNeeded := int(targetMemBytes / (arraySize * bytesPerEntry))
	
	if arraysNeeded < 1 {
		arraysNeeded = 1
	}
	
	if debug {
		logMessage(fmt.Sprintf("Memory test allocating %d arrays of %d elements each", arraysNeeded, arraySize), debug)
	}
	
	// Memory allocation
	var arrays [][]int64
	allocStart := time.Now()
	
	for i := 0; i < arraysNeeded; i++ {
		arr := make([]int64, arraySize)
		for j := range arr {
			arr[j] = rand.Int63()
		}
		arrays = append(arrays, arr)
		
		if debug && i%10 == 0 {
			logMessage(fmt.Sprintf("Memory allocation progress: %d/%d arrays", i+1, arraysNeeded), debug)
		}
	}
	
	allocDuration := time.Since(allocStart)
	bytesAllocated := uint64(arraysNeeded) * arraySize * bytesPerEntry
	allocSpeedMBps := float64(bytesAllocated) / allocDuration.Seconds() / (1024 * 1024)
	
	if debug {
		logMessage(fmt.Sprintf("Memory allocated: %.2f GB out of %.2f GB available (%.2f%% of system memory)",
			float64(bytesAllocated)/(1024*1024*1024),
			float64(totalMem)/(1024*1024*1024),
			float64(bytesAllocated)*100/float64(totalMem)), debug)
		logMessage(fmt.Sprintf("Memory write speed: %.2f MB/s", allocSpeedMBps), debug)
	}
	
	// Update performance stats
	perfStats.mu.Lock()
	perfStats.Memory.WriteSpeed = allocSpeedMBps
	perfStats.mu.Unlock()
	
	// Continuously read/write memory to ensure it stays allocated
	accessStart := time.Now()
	var bytesRead uint64 = 0
	readOps := 0
	
	for {
		select {
		case <-stop:
			// Calculate read performance
			readDuration := time.Since(accessStart)
			readSpeedMBps := float64(bytesRead) / readDuration.Seconds() / (1024 * 1024)
			
			perfStats.mu.Lock()
			perfStats.Memory.ReadSpeed = readSpeedMBps
			perfStats.mu.Unlock()
			
			if debug {
				logMessage(fmt.Sprintf("Memory read speed: %.2f MB/s (%d ops in %v)",
					readSpeedMBps, readOps, readDuration), debug)
			}
			return
		default:
			// Randomly access memory
			for i := 0; i < 1000; i++ {
				arrIdx := rand.Intn(len(arrays))
				elemIdx := rand.Intn(arraySize)
				
				// Read operation
				val := arrays[arrIdx][elemIdx]
				bytesRead += bytesPerEntry
				
				// Write operation (XOR toggle a bit)
				arrays[arrIdx][elemIdx] = val ^ 0xFF
				
				readOps++
			}
			
			// Periodically force GC to ensure memory doesn't get collected
			if rand.Intn(1_000) == 0 {
				runtime.GC()
			}
		}
	}
}

// --- 修正後的 performDiskWrite ---
func performDiskWrite(filePath string, data []byte, mode string, blockSize int64) (err error) {
    // 使用 os.OpenFile 以便更清晰地控制標誌
    file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
    if err != nil {
        return fmt.Errorf("failed to open/create file for writing (%s): %w", filePath, err)
    }
    
    // 不使用 defer 自動關閉，改為在每個錯誤處理和函數末尾手動關閉

    totalSize := int64(len(data))

    if mode == "sequential" {
        // --- 順序寫入 ---
        bytesWritten := int64(0)
        for bytesWritten < totalSize {
            chunkSize := blockSize
            if totalSize-bytesWritten < chunkSize {
                chunkSize = totalSize - bytesWritten
            }
            n, writeErr := file.Write(data[bytesWritten : bytesWritten+chunkSize])
            if writeErr != nil {
                file.Close() // 錯誤前關閉文件
                return fmt.Errorf("sequential write error on %s at offset %d: %w", filePath, bytesWritten, writeErr)
            }
            if int64(n) != chunkSize {
                file.Close() // 錯誤前關閉文件
                return fmt.Errorf("sequential write short write on %s: wrote %d bytes, expected %d", filePath, n, chunkSize)
            }
            bytesWritten += int64(n)
        }
    } else {
        // --- 隨機寫入 ---
        blocks := totalSize / blockSize
        if totalSize%blockSize > 0 {
            blocks++
        }

        blockOrder := make([]int64, blocks)
        for i := int64(0); i < blocks; i++ {
            blockOrder[i] = i
        }
        source := rand.NewSource(time.Now().UnixNano())
        rng := rand.New(source)
        rng.Shuffle(int(blocks), func(i, j int) {
            blockOrder[i], blockOrder[j] = blockOrder[j], blockOrder[i]
        })

        for _, blockIdx := range blockOrder {
            start := blockIdx * blockSize
            end := start + blockSize
            if end > totalSize {
                end = totalSize
            }
            chunkSize := end - start
            if chunkSize <= 0 {
                continue
            }

            n, writeErr := file.WriteAt(data[start:end], start)
            if writeErr != nil {
                file.Close() // 錯誤前關閉文件
                return fmt.Errorf("random write error on %s at offset %d: %w", filePath, start, writeErr)
            }
            if int64(n) != chunkSize {
                file.Close() // 錯誤前關閉文件
                return fmt.Errorf("random write short write on %s at offset %d: wrote %d bytes, expected %d", filePath, start, n, chunkSize)
            }
        }
    }

    // --- 強制同步數據到磁碟 ---
    if syncErr := file.Sync(); syncErr != nil {
        file.Close() // 錯誤前關閉文件
        return fmt.Errorf("failed to sync file (%s) after writing: %w", filePath, syncErr)
    }

    // --- 正確關閉文件 ---
    if closeErr := file.Close(); closeErr != nil {
        return fmt.Errorf("failed to close file (%s) after writing and sync: %w", filePath, closeErr)
    }

    // 確保文件已完全寫入並可以被讀取 - 可選的額外驗證
    // 這一步是防止在寫入後讀取時發生問題
    if _, statErr := os.Stat(filePath); statErr != nil {
        return fmt.Errorf("file verification failed after writing (%s): %w", filePath, statErr)
    }

    return nil
}

// --- 修正後的 performDiskReadAndVerify ---
func performDiskReadAndVerify(filePath string, originalData []byte, mode string, blockSize int64) error {
    // 檢查文件是否存在且可訪問
    if _, statErr := os.Stat(filePath); statErr != nil {
        return fmt.Errorf("file not accessible for reading (%s): %w", filePath, statErr)
    }
    
    file, err := os.Open(filePath) // 只讀模式打開
    if err != nil {
        return fmt.Errorf("failed to open file for reading (%s): %w", filePath, err)
    }
    defer file.Close() // 確保文件被關閉

    // 獲取文件信息
    fileInfo, err := file.Stat()
    if err != nil {
        return fmt.Errorf("failed to get file info (%s): %w", filePath, err)
    }
    totalSize := fileInfo.Size()
    expectedSize := int64(len(originalData))

    // 文件大小驗證
    if totalSize != expectedSize {
        return fmt.Errorf("file size mismatch on %s: expected %d bytes, got %d bytes", filePath, expectedSize, totalSize)
    }

    // 準備讀取緩衝區
    readData := make([]byte, totalSize)

    if mode == "sequential" {
        // --- 順序讀取 ---
        // 改為一次性讀取整個文件，這樣更可靠
        n, readErr := io.ReadFull(file, readData)
        if readErr != nil && readErr != io.EOF {
            return fmt.Errorf("sequential read error on %s: %w", filePath, readErr)
        }
        if int64(n) != totalSize {
            return fmt.Errorf("sequential read size mismatch on %s: read %d bytes, expected %d", filePath, n, totalSize)
        }
    } else {
        // --- 隨機讀取 ---
        blocks := totalSize / blockSize
        if totalSize%blockSize > 0 {
            blocks++
        }

        blockOrder := make([]int64, blocks)
        for i := int64(0); i < blocks; i++ {
            blockOrder[i] = i
        }
        source := rand.NewSource(time.Now().UnixNano())
        rng := rand.New(source)
        rng.Shuffle(int(blocks), func(i, j int) {
            blockOrder[i], blockOrder[j] = blockOrder[j], blockOrder[i]
        })

        for _, blockIdx := range blockOrder {
            start := blockIdx * blockSize
            end := start + blockSize
            if end > totalSize {
                end = totalSize
            }
            chunkSize := end - start
            if chunkSize <= 0 {
                continue
            }

            n, readErr := file.ReadAt(readData[start:end], start)
            if readErr != nil && readErr != io.EOF {
                return fmt.Errorf("random read error on %s at offset %d: %w", filePath, start, readErr)
            }
            // 對於最後一個區塊，可能會有 EOF，但應該已讀取了所有數據
            if int64(n) != chunkSize {
                return fmt.Errorf("random read short read on %s at offset %d: read %d bytes, expected %d", filePath, start, n, chunkSize)
            }
        }
    }

    // --- 執行數據驗證 ---
    if !bytes.Equal(originalData, readData) {
        // 尋找第一個不匹配的字節
        var mismatchPos int64 = -1
        for i := range originalData {
            if i >= len(readData) || originalData[i] != readData[i] {
                mismatchPos = int64(i)
                break
            }
        }
        return fmt.Errorf("data verification failed for file %s: read data does not match original data (first mismatch at byte %d)", filePath, mismatchPos)
    }

    return nil
}

func diskTest(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, memIntegrationChan chan []byte, config DiskTestConfig, perfStats *PerformanceStats, debug bool) {
    defer wg.Done()

    if len(config.MountPoints) == 0 {
        config.MountPoints = []string{"."}
    }
    if config.FileSize <= 0 {
        config.FileSize = 10 * 1024 * 1024 // Default 10MB
    }
    if config.BlockSize <= 0 {
        config.BlockSize = 4 * 1024 // Default 4KB
    }

    // 處理 both 模式
    var testModes []string
    if config.TestMode == "both" {
        testModes = []string{"sequential", "random"}
    } else {
        testModes = []string{config.TestMode}
    }

    // 為每個掛載點啟動一個 goroutine
    for _, mountPoint := range config.MountPoints {
        wg.Add(1)
        go func(mp string) {
            defer wg.Done()

            // 創建測試數據 (在循環外創建一次即可)
            data := make([]byte, config.FileSize)
            if _, err := rand.Read(data); err != nil {
                errorMsg := fmt.Sprintf("Failed to generate random data for %s: %v", mp, err)
                errorChan <- errorMsg
                logMessage(errorMsg, true) // Log critical error
                return
            }

            // 內存集成 (可以放在循環外)
            select {
            case memIntegrationChan <- data:
                logMessage(fmt.Sprintf("Sent %s of data to memory integration from %s",
                           formatSize(config.FileSize), mp), debug)
            default:
                // Channel full or no receiver
            }

            // 針對 both 模式，需要對每種模式分別進行測試
            for _, mode := range testModes {
                filePath := filepath.Join(mp, fmt.Sprintf("stress_test_%s.dat", mode))
                logMessage(fmt.Sprintf("Starting disk test on mount point: %s (mode: %s, size: %s, block: %s)",
                          mp, mode, formatSize(config.FileSize), formatSize(config.BlockSize)), debug)

                diskPerf := DiskPerformance{
                    MountPoint: mp,
                    Mode:       mode,
                    BlockSize:  config.BlockSize,
                }

                for {
                    select {
                    case <-stop:
                        os.Remove(filePath) // 清理測試文件
                        logMessage(fmt.Sprintf("Disk test stopped on %s (mode: %s)", mp, mode), debug)
                        return
                    default:
                        // --- 寫入測試 ---
                        writeStart := time.Now()
                        if err := performDiskWrite(filePath, data, mode, config.BlockSize); err != nil {
                            errorMsg := fmt.Sprintf("Disk write error on %s (mode: %s): %v", mp, mode, err)
                            errorChan <- errorMsg
                            logMessage(errorMsg, true) // Log error
                            time.Sleep(1 * time.Second) // 發生錯誤時稍等久一點再重試
                            continue
                        }
                        writeDuration := time.Since(writeStart)
                        writeSpeedMBps := float64(config.FileSize) / writeDuration.Seconds() / (1024 * 1024)
                        logMessage(fmt.Sprintf("Disk write on %s (mode: %s): %.2f MB/s (%s in %v)",
                                  mp, mode, writeSpeedMBps, formatSize(config.FileSize), writeDuration), debug)

                        // --- 讀取和驗證測試 ---
                        readStart := time.Now()
                        // 確保讀取和寫入使用相同的模式
                        if err := performDiskReadAndVerify(filePath, data, mode, config.BlockSize); err != nil {
                            errorMsg := fmt.Sprintf("Disk read/verify error on %s (mode: %s): %v", mp, mode, err)
                            errorChan <- errorMsg
                            logMessage(errorMsg, true) // Log error
                            time.Sleep(1 * time.Second) // 發生錯誤時稍等久一點再重試
                            continue
                        }
                        readDuration := time.Since(readStart)
                        readSpeedMBps := float64(config.FileSize) / readDuration.Seconds() / (1024 * 1024)
                        logMessage(fmt.Sprintf("Disk read/verify on %s (mode: %s): %.2f MB/s (%s in %v)",
                                  mp, mode, readSpeedMBps, formatSize(config.FileSize), readDuration), debug)

                        // --- 更新性能統計 ---
                        perfStats.mu.Lock()
                        diskPerf.ReadSpeed = readSpeedMBps
                        diskPerf.WriteSpeed = writeSpeedMBps

                        found := false
                        for i, dp := range perfStats.Disk {
                            if dp.MountPoint == mp && dp.Mode == mode && dp.BlockSize == config.BlockSize {
                                // 只更新最佳性能
                                if readSpeedMBps > dp.ReadSpeed {
                                    perfStats.Disk[i].ReadSpeed = readSpeedMBps
                                }
                                if writeSpeedMBps > dp.WriteSpeed {
                                    perfStats.Disk[i].WriteSpeed = writeSpeedMBps
                                }
                                found = true
                                break
                            }
                        }
                        if !found {
                            newPerf := diskPerf
                            perfStats.Disk = append(perfStats.Disk, newPerf)
                        }
                        perfStats.mu.Unlock()

                        // 短暫延遲避免 CPU/Disk 滿載
                        time.Sleep(100 * time.Millisecond)
                    }
                }
            }
        }(mountPoint)
    }
}

// Logger function to handle both console output and file logging
func logMessage(message string, debug bool) {
	timestamp := time.Now().Format("2006-01-02 15:04:05")
	logEntry := fmt.Sprintf("%s | %s", timestamp, message)
	
	// Always log to file
	f, err := os.OpenFile("stress.log", os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err == nil {
		defer f.Close()
		logger := log.New(f, "", 0)
		logger.Println(logEntry)
	}
	
	// Print to console only in debug mode or if explicitly requested
	if debug {
		fmt.Println(logEntry)
	}
}

// Load configuration from file
func loadConfig() (Config, error) {
	var config Config
	config.Debug = false // Default value
	
	data, err := os.ReadFile("config.json")
	if err != nil {
		return config, err
	}
	
	err = json.Unmarshal(data, &config)
	return config, err
}

// formatSize converts bytes to human-readable string (KB, MB, GB)
func formatSize(size int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	
	if size >= GB {
		return fmt.Sprintf("%.2fGB", float64(size)/float64(GB))
	}
	if size >= MB {
		return fmt.Sprintf("%.2fMB", float64(size)/float64(MB))
	}
	if size >= KB {
		return fmt.Sprintf("%.2fKB", float64(size)/float64(KB))
	}
	
	return fmt.Sprintf("%dB", size)
}

// parseSize parses size string with units (e.g., 4K, 64K, 1G)
func parseSize(sizeStr string) (int64, error) {
	sizeStr = strings.ToUpper(sizeStr)
	var multiplier int64 = 1
	
	if strings.HasSuffix(sizeStr, "K") {
		multiplier = 1024
		sizeStr = sizeStr[:len(sizeStr)-1]
	} else if strings.HasSuffix(sizeStr, "KB") {
		multiplier = 1024
		sizeStr = sizeStr[:len(sizeStr)-2]
	} else if strings.HasSuffix(sizeStr, "M") {
		multiplier = 1024 * 1024
		sizeStr = sizeStr[:len(sizeStr)-1]
	} else if strings.HasSuffix(sizeStr, "MB") {
		multiplier = 1024 * 1024
		sizeStr = sizeStr[:len(sizeStr)-2]
	} else if strings.HasSuffix(sizeStr, "G") {
		multiplier = 1024 * 1024 * 1024
		sizeStr = sizeStr[:len(sizeStr)-1]
	} else if strings.HasSuffix(sizeStr, "GB") {
		multiplier = 1024 * 1024 * 1024
		sizeStr = sizeStr[:len(sizeStr)-2]
	}
	
	size, err := strconv.ParseInt(sizeStr, 10, 64)
	if err != nil {
		return 0, err
	}
	
	return size * multiplier, nil
}

func main() {
	// Parse command line arguments
	var mountPoints string
	var fileSize string
	var testMode string
	var blockSizes string
	var duration string
	var testCPU bool
	var testMemory bool
	var showHelp bool

	flag.StringVar(&mountPoints, "l", "", "Comma separated mount points to test (e.g. /mnt/disk1,/mnt/disk2)")
	flag.StringVar(&fileSize, "size", "10MB", "Size of test files (supports K, M, G units)")
	flag.StringVar(&testMode, "mode", "both", "Test mode: sequential, random, or both")
	flag.StringVar(&blockSizes, "block", "4K", "Comma separated block sizes for disk operations (supports K, M, G units)")
	flag.StringVar(&duration, "duration", "10m", "Test duration (e.g. 30s, 5m, 1h)")
	flag.BoolVar(&testCPU, "cpu", false, "Enable CPU testing")
	flag.BoolVar(&testMemory, "memory", false, "Enable memory testing")
	flag.BoolVar(&showHelp, "h", false, "Show help")
	flag.Parse()

	// Check if any tests are enabled, otherwise show help
	if (!testCPU && !testMemory && mountPoints == "") || showHelp {
		fmt.Println("System Stress Test Tool")
		fmt.Println("Usage: stress [options]")
		fmt.Println("\nOptions:")
		flag.PrintDefaults()
		fmt.Println("\nAt least one of -cpu, -memory, or -l must be specified.")
		return
	}

	// Load configuration
	config, err := loadConfig()
	if err != nil {
		fmt.Printf("Failed to load config.json, using default settings: %v\n", err)
	}

	// Set up logging
	debug := config.Debug

	// Parse test duration
	testDuration, err := time.ParseDuration(duration)
	if err != nil {
		logMessage(fmt.Sprintf("Invalid duration format: %s, using default 10 minutes", duration), true)
		testDuration = 10 * time.Minute
	}

	logMessage(fmt.Sprintf("Starting stress test for %v...", testDuration), true)

	// Performance stats tracking
	perfStats := &PerformanceStats{
		CPU:    CPUPerformance{},
		Memory: MemoryPerformance{},
		Disk:   []DiskPerformance{},
	}

	var wg sync.WaitGroup
	stop := make(chan struct{})
	errorChan := make(chan string, 100) // Buffer for error messages
	memIntegrationChan := make(chan []byte, 10) // Channel for memory-disk integration

	// Track test results
	results := TestResult{
		CPU:  "PASS",
		DIMM: "PASS",
		HDD:  "PASS",
	}

	// Error collection map for detailed reporting
	errorDetails := make(map[string][]string)

	// Get NUMA node information
	numaInfo, err := getNUMAInfo()
	if err != nil {
		logMessage(fmt.Sprintf("Failed to get NUMA info: %v, falling back to single node", err), debug)
	}

	if debug {
		logMessage(fmt.Sprintf("Detected %d NUMA nodes", numaInfo.NumNodes), debug)
		for i, cpus := range numaInfo.NodeCPUs {
			if len(cpus) > 0 {
				logMessage(fmt.Sprintf("NUMA node %d has CPUs: %v", i, cpus), debug)
			}
		}
	}

	// Run memory and disk tests first, as requested
	if testMemory {
		memUsageTarget := 0.7 // 70% of free memory usage target
		logMessage("Starting memory stress test...", debug)
		wg.Add(1)
		go memoryTest(&wg, stop, errorChan, memUsageTarget, perfStats, debug)
	}

	// Parse mount points for disk test
	var mounts []string
	if mountPoints != "" {
		mounts = strings.Split(mountPoints, ",")
		logMessage(fmt.Sprintf("Starting disk tests on mount points: %v", mounts), debug)
	}

	// Parse file size
	fileSizeBytes, err := parseSize(fileSize)
	if err != nil {
		logMessage(fmt.Sprintf("Invalid file size: %v, using default 10MB", err), true)
		fileSizeBytes = 10 * 1024 * 1024
	}

	// Parse test mode
	var testModes []string
	switch testMode {
	case "sequential":
		testModes = []string{"sequential"}
	case "random":
		testModes = []string{"random"}
	case "both", "":
		testModes = []string{"sequential", "random"}
	default:
		logMessage(fmt.Sprintf("Invalid test mode: %s, using both sequential and random", testMode), true)
		testModes = []string{"sequential", "random"}
	}

	// Parse block size list
	var blockSizeList []int64
	if blockSizes != "" {
		for _, bsStr := range strings.Split(blockSizes, ",") {
			bs, err := parseSize(bsStr)
			if err != nil {
				logMessage(fmt.Sprintf("Invalid block size: %v, skipping", err), debug)
				continue
			}
			blockSizeList = append(blockSizeList, bs)
		}
	}
	if len(blockSizeList) == 0 {
		// If no valid block sizes, use default
		blockSizeList = append(blockSizeList, 4*1024) // 4K default
	}

	// Run disk tests if mount points are specified
	if len(mounts) > 0 {
		for _, mode := range testModes {
			for _, bs := range blockSizeList {
				// Format block size for display
				bsDisplay := formatSize(bs)
				logMessage(fmt.Sprintf("Starting %s disk test with file size %s and block size %s on %v...",
					mode, formatSize(fileSizeBytes), bsDisplay, mounts), debug)

				diskConfig := DiskTestConfig{
					MountPoints: mounts,
					FileSize:    fileSizeBytes,
					TestMode:    mode,
					BlockSize:   bs,
				}

				wg.Add(1)
				go diskTest(&wg, stop, errorChan, memIntegrationChan, diskConfig, perfStats, debug)
			}
		}
	}

	// Start CPU tests if enabled
	if testCPU {
		logMessage(fmt.Sprintf("Starting CPU stress tests using %d cores...", runtime.NumCPU()), debug)

		// Distribute CPU tasks across NUMA nodes
		for nodeIdx, cpus := range numaInfo.NodeCPUs {
			if len(cpus) == 0 {
				continue
			}

			logMessage(fmt.Sprintf("Starting CPU tests on NUMA node %d with %d CPUs", nodeIdx, len(cpus)), debug)

			// Distribute different computation types across CPUs in this NUMA node
			for i, cpuID := range cpus {
				testType := i % 3 // 0=integer, 1=float, 2=vector

				switch testType {
				case 0:
					wg.Add(1)
					go integerComputation(&wg, stop, errorChan, nodeIdx, cpuID, perfStats, debug)
				case 1:
					wg.Add(1)
					go floatComputation(&wg, stop, errorChan, nodeIdx, cpuID, perfStats, debug)
				case 2:
					wg.Add(1)
					go vectorComputation(&wg, stop, errorChan, nodeIdx, cpuID, perfStats, debug)
				}
			}
		}
	}

	// Error collection goroutine
	go func() {
		for err := range errorChan {
			if err == "" {
				continue
			}

			// Determine error type and update results
			switch {
			case strings.Contains(err, "Integer overflow") || strings.Contains(err, "Float") || strings.Contains(err, "Vector computation"):
				results.CPU = "FAIL"
				errorDetails["CPU"] = append(errorDetails["CPU"], err)
			case strings.Contains(err, "Memory"):
				results.DIMM = "FAIL"
				errorDetails["DIMM"] = append(errorDetails["DIMM"], err)
			case strings.Contains(err, "Disk"):
				results.HDD = "FAIL"
				errorDetails["HDD"] = append(errorDetails["HDD"], err)
			}

			logMessage(fmt.Sprintf("Error detected: %s", err), debug)
		}
	}()

	// Memory-disk integration goroutine
	go func() {
		for data := range memIntegrationChan {
			if testMemory {
				if debug {
					logMessage(fmt.Sprintf("Processing %s of data in memory integration", formatSize(int64(len(data)))), debug)
				}

				// Simulate processing the data in memory
				checksum := uint64(0)
				for i := 0; i < len(data); i += 8 {
					if i+8 <= len(data) {
						val := binary(data[i:i+8])
						checksum ^= val
					}
				}

				if debug {
					logMessage(fmt.Sprintf("Memory integration checksum: %X", checksum), debug)
				}
			}
		}
	}()

	// Setup progress reporting
	progressTicker := time.NewTicker(30 * time.Second)
	go func() {
		for {
			select {
			case <-progressTicker.C:
				// Snapshot current performance stats
				perfStats.mu.Lock()
				cpuGFLOPS := perfStats.CPU.GFLOPS
				memRead := perfStats.Memory.ReadSpeed
				memWrite := perfStats.Memory.WriteSpeed

				// Get best disk performance for reporting
				var bestDiskRead, bestDiskWrite float64
				var bestDiskMount string
				for _, disk := range perfStats.Disk {
					if disk.ReadSpeed > bestDiskRead {
						bestDiskRead = disk.ReadSpeed
						bestDiskMount = disk.MountPoint
					}
					if disk.WriteSpeed > bestDiskWrite {
						bestDiskWrite = disk.WriteSpeed
					}
				}
				perfStats.mu.Unlock()

				// Report progress
				progressMsg := fmt.Sprintf("Progress update - CPU: %.2f GFLOPS (approximate value, not exact)", cpuGFLOPS)
				if testMemory {
					progressMsg += fmt.Sprintf(", Memory: R=%.2f MB/s W=%.2f MB/s", memRead, memWrite)
				}
				if len(mounts) > 0 {
					progressMsg += fmt.Sprintf(", Disk(%s): R=%.2f MB/s W=%.2f MB/s",
						bestDiskMount, bestDiskRead, bestDiskWrite)
				}
				logMessage(progressMsg, true)

			case <-stop:
				progressTicker.Stop()
				return
			}
		}
	}()

	// Start time measurement
	startTime := time.Now()

	// Run for specified duration
	time.Sleep(testDuration)

	// Stop the test
	close(stop)
	close(memIntegrationChan)

	// Wait for all tests to finish
	wg.Wait()
	close(errorChan)

	elapsedTime := time.Since(startTime)

	// Log final performance results
	logMessage("=== PERFORMANCE RESULTS ===", true)

	// CPU performance
	if testCPU {
		logMessage(fmt.Sprintf("CPU Performance: %.2f GFLOPS", perfStats.CPU.GFLOPS), true)
	}

	// Memory performance
	if testMemory {
		logMessage(fmt.Sprintf("Memory Performance - Read: %.2f MB/s, Write: %.2f MB/s",
			perfStats.Memory.ReadSpeed, perfStats.Memory.WriteSpeed), true)
	}

	// Disk performance
	if len(mounts) > 0 {
		logMessage("Disk Performance:", true)
		for _, disk := range perfStats.Disk {
			logMessage(fmt.Sprintf("  Mount: %s, Mode: %s, Block: %s - Read: %.2f MB/s, Write: %.2f MB/s",
				disk.MountPoint, disk.Mode, formatSize(disk.BlockSize), disk.ReadSpeed, disk.WriteSpeed), true)
		}
	}

	// Log final results
	resultStr := fmt.Sprintf("Stress Test Summary - Duration: %s", elapsedTime.Round(time.Second))
	if testCPU {
		resultStr += fmt.Sprintf(" | CPU: %s", results.CPU)
	}
	if testMemory {
		resultStr += fmt.Sprintf(" | DIMM: %s", results.DIMM)
	}
	if len(mounts) > 0 {
		resultStr += fmt.Sprintf(" | HDD: %s", results.HDD)
	}

	// Add failure details if any
	for component, errors := range errorDetails {
		if len(errors) > 0 {
			resultStr += fmt.Sprintf("\n%s FAIL reason: %s", component, errors[0])
			if len(errors) > 1 {
				resultStr += fmt.Sprintf(" (and %d more errors)", len(errors)-1)
			}
		}
	}

	logMessage(resultStr, true) // Always show final results
	logMessage("Stress test completed!", true)
}

// binary converts a byte slice to a uint64, used for checksum calculation
func binary(buf []byte) uint64 {
	return uint64(buf[0]) | uint64(buf[1])<<8 | uint64(buf[2])<<16 | uint64(buf[3])<<24 |
		uint64(buf[4])<<32 | uint64(buf[5])<<40 | uint64(buf[6])<<48 | uint64(buf[7])<<56
}
