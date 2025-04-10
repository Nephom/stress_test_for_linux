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
	RandomAccessSpeed float64 // 新增
    UsagePercent     float64  // 新增
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

// Enhanced memory test with improved speed measurement
func memoryTest(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, memUsagePercent float64, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()

	totalMem, _ := getSystemMemory()

	// 使用參數指定的百分比，但針對總記憶體而非可用記憶體
	targetMemBytes := uint64(float64(totalMem) * memUsagePercent)

	if debug {
		logMessage(fmt.Sprintf("Memory test targeting %.2f%% of total system memory (%.2f GB)",
			memUsagePercent*100, float64(targetMemBytes)/(1024*1024*1024)), debug)
	}

	// 首先進行速度測試，避免大量記憶體分配影響測量結果
	// 使用單獨的中等大小數組進行速度測試
	const speedTestSize = 50_000_000 // 5千萬個int64，約400MB
	var speedTestArray []int64

	// 分配測試數組
	speedTestArray = make([]int64, speedTestSize)

	// 1. 順序寫入測試
	if debug {
		logMessage("Measuring sequential memory write speed...", debug)
	}

	writeStart := time.Now()
	for i := 0; i < speedTestSize; i++ {
		speedTestArray[i] = int64(i)
	}
	writeDuration := time.Since(writeStart)
	bytesWritten := speedTestSize * 8 // 每個int64是8字節
	writeSpeedMBps := float64(bytesWritten) / writeDuration.Seconds() / (1024 * 1024)

	// 2. 順序讀取測試
	if debug {
		logMessage("Measuring sequential memory read speed...", debug)
	}

	var sum int64 = 0 // 防止編譯器優化掉讀取操作
	readStart := time.Now()
	for i := 0; i < speedTestSize; i++ {
		sum += speedTestArray[i]
	}
	readDuration := time.Since(readStart)
	bytesRead := speedTestSize * 8
	readSpeedMBps := float64(bytesRead) / readDuration.Seconds() / (1024 * 1024)

	// 3. 隨機存取測試（讀+寫）
	if debug {
		logMessage("Measuring random memory access speed...", debug)
	}

	// 預生成隨機索引以排除隨機數生成的時間
	randIndices := make([]int, 5_000_000) // 5百萬個隨機訪問
	for i := range randIndices {
		randIndices[i] = rand.Intn(speedTestSize)
	}

	randomStart := time.Now()
	for _, idx := range randIndices {
		// 讀取然後寫入
		val := speedTestArray[idx]
		speedTestArray[idx] = val ^ 0x1
	}
	randomDuration := time.Since(randomStart)
	randomBytes := int64(len(randIndices) * 16) // 每次操作8字節讀 + 8字節寫
	randomSpeedMBps := float64(randomBytes) / randomDuration.Seconds() / (1024 * 1024)

	if debug {
		logMessage(fmt.Sprintf("Memory speed results:"), debug)
		logMessage(fmt.Sprintf("  - Sequential write: %.2f MB/s", writeSpeedMBps), debug)
		logMessage(fmt.Sprintf("  - Sequential read:  %.2f MB/s", readSpeedMBps), debug)
		logMessage(fmt.Sprintf("  - Random access:    %.2f MB/s", randomSpeedMBps), debug)
	}

	// 釋放速度測試數組
	speedTestArray = nil
	runtime.GC()

	// 現在開始實際記憶體分配測試
	const arraySize = 10_000_000 // 1千萬個int64
	const bytesPerEntry = 8
	arraysNeeded := int(targetMemBytes / (arraySize * bytesPerEntry))

	if arraysNeeded < 1 {
		arraysNeeded = 1
	}

	if debug {
		logMessage(fmt.Sprintf("Memory test allocating %d arrays of %d elements each", arraysNeeded, arraySize), debug)
	}

	// 記憶體分配
	var arrays [][]int64
	allocStart := time.Now()

	// 跟踪實際分配量
	var bytesAllocated uint64 = 0

	// 使用更健壯的分配方式與重試邏輯
	for i := 0; i < arraysNeeded; i++ {
		// 嘗試分配數組
		var arr []int64

		// 使用恢復機制處理可能的內存不足情況
		func() {
			defer func() {
				if r := recover(); r != nil && debug {
					logMessage(fmt.Sprintf("Recovered from allocation panic: %v", r), debug)
				}
			}()

			arr = make([]int64, arraySize)
			// 實際接觸記憶體確保物理分配
			for j := range arr {
				arr[j] = rand.Int63()
			}
		}()

		// 如果分配成功，添加到數組列表
		if arr != nil {
			arrays = append(arrays, arr)
			bytesAllocated += arraySize * bytesPerEntry

			if debug && (i+1)%10 == 0 {
				allocPercent := float64(bytesAllocated) * 100 / float64(totalMem)
				logMessage(fmt.Sprintf("Memory allocation progress: %d/%d arrays (%.2f%% of system memory)",
					i+1, arraysNeeded, allocPercent), debug)
			}
		} else {
			// 如果分配失敗，記錄並繼續
			if debug {
				logMessage("Failed to allocate memory array, continuing with what we have", debug)
			}
			break
		}

		// 檢查是否已分配足夠
		if bytesAllocated >= targetMemBytes {
			if debug {
				logMessage("Reached target memory allocation", debug)
			}
			break
		}

		// 定期強制垃圾回收以防止過早OOM
		if i%100 == 0 {
			runtime.GC()
		}
	}

	allocDuration := time.Since(allocStart)
	allocSpeedMBps := float64(bytesAllocated) / allocDuration.Seconds() / (1024 * 1024)
	memoryUsagePercent := float64(bytesAllocated) * 100 / float64(totalMem)

	if debug {
		logMessage(fmt.Sprintf("Memory allocated: %.2f GB out of %.2f GB total (%.2f%% of system memory)",
			float64(bytesAllocated)/(1024*1024*1024),
			float64(totalMem)/(1024*1024*1024),
			memoryUsagePercent), debug)
		logMessage(fmt.Sprintf("Memory bulk allocation speed: %.2f MB/s", allocSpeedMBps), debug)
	}

	// 更新性能統計
	perfStats.mu.Lock()
	perfStats.Memory.WriteSpeed = writeSpeedMBps
	perfStats.Memory.ReadSpeed = readSpeedMBps
	perfStats.Memory.RandomAccessSpeed = randomSpeedMBps
	perfStats.Memory.UsagePercent = memoryUsagePercent
	perfStats.mu.Unlock()

	// 檢查是否分配了足夠的內存
	if memoryUsagePercent < memUsagePercent*75.0 { // 如果達不到目標的75%則報錯
		errorChan <- fmt.Sprintf("Could only allocate %.2f%% of system memory, wanted %.2f%%",
			memoryUsagePercent, memUsagePercent*100)
	}

	// 持續讀寫內存以確保它保持分配
	for {
		select {
		case <-stop:
			if debug {
				logMessage("Memory test stopped.", debug)
			}
			return
		default:
			// 持續訪問內存
			for i := 0; i < 1000; i++ {
				if len(arrays) == 0 {
					break
				}

				arrIdx := rand.Intn(len(arrays))
				elemIdx := rand.Intn(arraySize)

				// 讀寫操作，確保內存保持活躍
				val := arrays[arrIdx][elemIdx]
				arrays[arrIdx][elemIdx] = val ^ 0xFF
			}

			// 偶爾強制GC以確保內存不被回收
			if rand.Intn(10_000) == 0 {
				runtime.GC()
			}

			// 添加短暫休眠以減少CPU使用
			time.Sleep(time.Millisecond)
		}
	}
}

// --- performDiskWrite (保持不變，因為邏輯看起來是健壯的) ---
func performDiskWrite(filePath string, data []byte, mode string, blockSize int64) (err error) {
	// 檢查文件是否已存在，如存在則先移除 (保持健壯性)
	// 注意：在高並發或權限受限環境下，這裡可能需要更細緻的錯誤處理或重試
	if _, statErr := os.Stat(filePath); statErr == nil {
		if rmErr := os.Remove(filePath); rmErr != nil {
			// 如果移除失敗，可能表示權限問題或文件被鎖定，這本身就是一個需要報告的問題
			return fmt.Errorf("failed to remove existing file (%s): %w", filePath, rmErr)
		}
	} else if !os.IsNotExist(statErr) {
        // 如果 Stat 返回的不是 "不存在" 錯誤，也報告它
        return fmt.Errorf("failed to stat file before write (%s): %w", filePath, statErr)
    }


	// 檢查資料長度
	if len(data) == 0 {
		return fmt.Errorf("attempt to write empty data to file %s", filePath)
	}

	// 使用 os.OpenFile 以便更清晰地控制標誌
	file, err := os.OpenFile(filePath, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
	if err != nil {
		return fmt.Errorf("failed to open/create file for writing (%s): %w", filePath, err)
	}
	// **注意：** 這裡開始，任何 return 都必須確保 file 被關閉

	totalSize := int64(len(data))
	var totalBytesWritten int64 = 0

	if mode == "sequential" {
		// --- 順序寫入 ---
		n, writeErr := file.Write(data)
		totalBytesWritten = int64(n) // 記錄實際寫入量
		if writeErr != nil {
			file.Close() // 錯誤前關閉文件
			return fmt.Errorf("sequential write error on %s after writing %d bytes: %w", filePath, totalBytesWritten, writeErr)
		}
		if totalBytesWritten != totalSize {
			file.Close() // 錯誤前關閉文件
			return fmt.Errorf("sequential write short write on %s: wrote %d bytes, expected %d", filePath, totalBytesWritten, totalSize)
		}
	} else { // mode == "random"
		// --- 隨機寫入 ---
		// 首先填充文件到正確的大小 (Truncate)
        // 注意: Truncate 在某些系統/FS 上可能只是邏輯擴展，不一定物理分配，但 WriteAt 會觸發物理分配
		if err := file.Truncate(totalSize); err != nil {
			file.Close()
			return fmt.Errorf("failed to truncate file to required size %d for (%s): %w", totalSize, filePath, err)
		}

		blocks := totalSize / blockSize
		if totalSize%blockSize > 0 {
			blocks++
		}

		blockOrder := make([]int64, blocks)
		for i := int64(0); i < blocks; i++ {
			blockOrder[i] = i
		}
        // 使用確定性的隨機源可能更有利於調試，但 time.Now() 用於壓力測試是常見的
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
				continue // 理論上不應該發生，除非 totalSize=0 或 blockSize <= 0 (已被外部檢查過)
			}

            if start >= int64(len(data)) || end > int64(len(data)) {
                file.Close()
                return fmt.Errorf("internal logic error: calculated range [%d:%d] exceeds data length %d", start, end, len(data))
            }

			n, writeErr := file.WriteAt(data[start:end], start)
			totalBytesWritten += int64(n) // 累加實際寫入量
			if writeErr != nil {
				file.Close() // 錯誤前關閉文件
				return fmt.Errorf("random write error on %s at offset %d after writing %d bytes for this chunk: %w", filePath, start, n, writeErr)
			}
			if int64(n) != chunkSize {
				file.Close() // 錯誤前關閉文件
				return fmt.Errorf("random write short write on %s at offset %d: wrote %d bytes, expected %d", filePath, start, n, chunkSize)
			}
		}
         // 隨機寫入後，再次檢查總寫入量 (雙重保險)
        if totalBytesWritten != totalSize {
            file.Close()
            return fmt.Errorf("random write total bytes written mismatch: wrote %d bytes, expected %d for %s", totalBytesWritten, totalSize, filePath)
        }
	}

	// --- 強制同步數據到磁碟 ---
	if syncErr := file.Sync(); syncErr != nil {
		file.Close() // 錯誤前關閉文件
		// Sync 失敗是嚴重問題，可能表示磁碟問題或權限不足
		return fmt.Errorf("failed to sync file (%s) after writing %d bytes: %w", filePath, totalBytesWritten, syncErr)
	}

	// --- 正確關閉文件 ---
	if closeErr := file.Close(); closeErr != nil {
		// 即使 Sync 成功，Close 也可能失敗 (雖然少見)
		return fmt.Errorf("failed to close file (%s) after writing and sync: %w", filePath, closeErr)
	}

	// --- 最終驗證：文件系統報告的大小 ---
	// 這是寫操作成功的最後一道關卡
	fileInfo, statErr := os.Stat(filePath)
	if statErr != nil {
		// 如果 Stat 失敗，即使前面都成功，也意味著有問題
		return fmt.Errorf("final file verification failed after close (%s): cannot stat file: %w", filePath, statErr)
	}

	// 檢查文件大小
	if fileInfo.Size() != totalSize {
		// 如果大小不匹配，說明寫入操作最終沒有按預期完成
		return fmt.Errorf("final file size verification failed: expected %d bytes, got %d bytes for file %s",
			totalSize, fileInfo.Size(), filePath)
	}

	// 所有檢查通過，返回 nil
	return nil
}


// --- 修正後的 performDiskReadAndVerify ---
func performDiskReadAndVerify(filePath string, originalData []byte, mode string, blockSize int64) error {
	expectedSize := int64(len(originalData))

	// 1. 檢查文件是否存在且可訪問，並獲取大小
	fileInfo, statErr := os.Stat(filePath)
	if statErr != nil {
		if os.IsNotExist(statErr) {
			return fmt.Errorf("file not found for reading (%s): %w", filePath, statErr)
		}
		return fmt.Errorf("file not accessible for reading (%s): cannot stat file: %w", filePath, statErr)
	}

	// 2. 文件大小驗證 (讀取前的第一道關卡)
	totalSize := fileInfo.Size()
	if totalSize != expectedSize {
		return fmt.Errorf("file size mismatch before reading %s: expected %d bytes, found %d bytes", filePath, expectedSize, totalSize)
	}
	// 如果預期大小為 0，無需讀取，直接成功
	if expectedSize == 0 {
		return nil
	}

	// 3. 打開文件進行讀取
	file, err := os.Open(filePath) // 只讀模式打開
	if err != nil {
		return fmt.Errorf("failed to open file for reading (%s): %w", filePath, err)
	}
	defer file.Close() // 確保文件最終被關閉

	// 4. 準備讀取緩衝區
	// 使用 stat 獲取的大小，而不是 len(originalData)，雖然它們此時應該相等
	readData := make([]byte, totalSize)

	if mode == "sequential" {
		// --- 順序讀取 ---
		// 使用 io.ReadFull 來確保讀滿緩衝區
		n, readErr := io.ReadFull(file, readData)

		// 檢查 ReadFull 的結果
		if readErr != nil {
			// 如果錯誤是 io.ErrUnexpectedEOF，說明文件比 Stat 報告的要短，這是一個嚴重的不一致
			if readErr == io.ErrUnexpectedEOF {
				return fmt.Errorf("sequential read error on %s: file was shorter than expected (read %d bytes, expected %d). Inconsistency detected: %w", filePath, n, totalSize, readErr)
			}
			// 其他任何錯誤都是讀取失敗
			return fmt.Errorf("sequential read error on %s after reading %d bytes: %w", filePath, n, readErr)
		}
		// 如果 readErr 為 nil，ReadFull 保證 n == len(readData) == totalSize
		// 下面的檢查理論上是多餘的，但保留作為額外的斷言
		if int64(n) != totalSize {
			return fmt.Errorf("internal inconsistency: ReadFull returned nil error but read %d bytes, expected %d for %s", n, totalSize, filePath)
		}
	} else { // mode == "random"
		// --- 隨機讀取 ---
		// 與寫入時相同的塊計算和隨機化邏輯
		blocks := totalSize / blockSize
		if totalSize%blockSize > 0 {
			blocks++
		}

		blockOrder := make([]int64, blocks)
		for i := int64(0); i < blocks; i++ {
			blockOrder[i] = i
		}
		// 注意：為了驗證，讀取的隨機順序最好與寫入時不同，或者使用固定的隨機種子
        // 但對於壓力測試，每次不同也可以接受
		source := rand.NewSource(time.Now().UnixNano() + 1) // 加 1 避免與寫入完全相同
		rng := rand.New(source)
		rng.Shuffle(int(blocks), func(i, j int) {
			blockOrder[i], blockOrder[j] = blockOrder[j], blockOrder[i]
		})

        var totalBytesRead int64 = 0
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

            // 確保讀取的切片範圍有效
            if start >= int64(len(readData)) || end > int64(len(readData)) {
                 return fmt.Errorf("internal logic error during random read: calculated range [%d:%d] exceeds buffer length %d", start, end, len(readData))
            }

			// 從文件讀取到 readData 的 *對應位置*
			n, readErr := file.ReadAt(readData[start:end], start)
            totalBytesRead += int64(n)

			// 檢查短讀 (最常見的錯誤)
			if int64(n) != chunkSize {
				// 即使錯誤是 EOF，只要讀到的字節數不等於期望的 chunkSize，就是有問題
				// (除非 chunkSize 為 0，但前面已經 continue 了)
				return fmt.Errorf("random read short read on %s at offset %d: read %d bytes, expected %d (error: %v)", filePath, start, n, chunkSize, readErr)
			}

			// 檢查讀取錯誤 (排除預期的 EOF)
			// 如果 n == chunkSize，那麼 err 只能是 nil 或 io.EOF (當且僅當讀取的剛好是文件的最後一個字節)
			if readErr != nil && readErr != io.EOF {
				return fmt.Errorf("random read error on %s at offset %d after reading %d bytes for this chunk: %w", filePath, start, n, readErr)
			}
		}
        // 隨機讀取後，可以選擇性地再檢查一次總讀取量
        // 這有助於捕捉 block 計算或循環中的邏輯錯誤
        // 注意：這裡的 totalBytesRead 可能不等於 totalSize，因為塊是隨機讀取的，
        // 我們關心的是每個塊是否讀取正確，以及最終數據是否匹配。
        // 因此，這個檢查可能不是很有用，除非你想驗證塊計算邏輯。
        /*
        if totalBytesRead != totalSize { // 這個檢查可能不適用於隨機讀取驗證
             return fmt.Errorf("random read total bytes read mismatch: read %d bytes, expected %d for %s", totalBytesRead, totalSize, filePath)
        }
        */
	}

	// --- 5. 執行數據驗證 ---
	if !bytes.Equal(originalData, readData) {
		// 如果數據不匹配，查找第一個差異點以提供更多信息
		mismatchPos := int64(-1)
		var originalByte, readByte byte
		limit := len(originalData)
		if len(readData) < limit {
			limit = len(readData) // 防止比較時 readData 越界
		}

		for i := 0; i < limit; i++ {
			if originalData[i] != readData[i] {
				mismatchPos = int64(i)
				originalByte = originalData[i]
				readByte = readData[i]
				break
			}
		}
        // 如果循環結束還沒找到差異，但長度不同 (理論上已被前面的大小檢查攔截)
        if mismatchPos == -1 && len(originalData) != len(readData) {
            mismatchPos = int64(limit) // 差異發生在較短數據的末尾
            if len(originalData) > limit { originalByte = originalData[limit] }
            // readByte 保持 0 或默認值
        }

		return fmt.Errorf("data verification failed for file %s: read data does not match original data (lengths: original=%d, read=%d). First mismatch at byte %d (original: %d[0x%X], read: %d[0x%X])",
			filePath, len(originalData), len(readData), mismatchPos, originalByte, originalByte, readByte, readByte)
	}

	// 所有驗證通過
	return nil
}


// --- 修正後的 diskTest ---
func diskTest(wg *sync.WaitGroup, stop chan struct{}, errorChan chan string, config DiskTestConfig, perfStats *PerformanceStats, debug bool) {
	defer wg.Done()

	if len(config.MountPoints) == 0 {
		logMessage("No mount points specified, using current directory '.'", true)
		config.MountPoints = []string{"."}
	}
	if config.FileSize <= 0 {
		logMessage(fmt.Sprintf("Invalid FileSize %d, using default 10MB", config.FileSize), true)
		config.FileSize = 10 * 1024 * 1024 // Default 10MB
	}
	if config.BlockSize <= 0 {
		logMessage(fmt.Sprintf("Invalid BlockSize %d, using default 4KB", config.BlockSize), true)
		config.BlockSize = 4 * 1024 // Default 4KB
	}
	if config.FileSize < config.BlockSize && config.TestMode != "sequential" {
        // 對於隨機模式，文件大小至少要等於塊大小才有意義（儘管代碼能處理）
        logMessage(fmt.Sprintf("Warning: FileSize (%s) is smaller than BlockSize (%s) for random/both mode.", formatSize(config.FileSize), formatSize(config.BlockSize)), true)
    }


	// 處理 both 模式
	var testModes []string
	if config.TestMode == "both" {
		testModes = []string{"sequential", "random"}
	} else if config.TestMode == "sequential" || config.TestMode == "random" {
		testModes = []string{config.TestMode}
	} else {
		errorMsg := fmt.Sprintf("Invalid test mode: %s. Use 'sequential', 'random', or 'both'.", config.TestMode)
		errorChan <- errorMsg
		logMessage(errorMsg, true)
		return
	}

	// 為每個掛載點啟動一個 goroutine
	mountWg := &sync.WaitGroup{} // 用於等待所有掛載點的 goroutine 完成
	for _, mountPoint := range config.MountPoints {
		mountWg.Add(1)
		go func(mp string) {
			defer mountWg.Done()

			// 檢查掛載點是否存在且可寫入
			if info, err := os.Stat(mp); err != nil {
				errorMsg := fmt.Sprintf("Mount point %s not accessible: %v", mp, err)
				errorChan <- errorMsg
				logMessage(errorMsg, true)
				return
			} else if !info.IsDir() {
				errorMsg := fmt.Sprintf("Mount point %s is not a directory", mp)
				errorChan <- errorMsg
				logMessage(errorMsg, true)
				return
			}
            // 嘗試在掛載點創建一個臨時文件來檢查寫權限
            tempFilePath := filepath.Join(mp, fmt.Sprintf(".writetest_%d", time.Now().UnixNano()))
            tempFile, err := os.Create(tempFilePath)
            if err != nil {
                errorMsg := fmt.Sprintf("Mount point %s is not writable: %v", mp, err)
				errorChan <- errorMsg
				logMessage(errorMsg, true)
				return
            }
            tempFile.Close()
            os.Remove(tempFilePath) // 清理臨時文件


			// 檢查磁盤空間 (需要兩倍空間：一個用於寫入，一個用於可能的系統緩存/操作)
            // 這個檢查可能不完全準確，但作為預防措施
			var stat syscall.Statfs_t
            requiredSpace := uint64(config.FileSize) * 2 // 請求至少兩倍文件大小的空間
			if err := syscall.Statfs(mp, &stat); err == nil {
				availableBytes := stat.Bavail * uint64(stat.Bsize) // 可用空間 (非 root 用戶)
				if availableBytes < requiredSpace {
					errorMsg := fmt.Sprintf("Insufficient disk space on %s: required approx %s, available %s",
						mp, formatSize(int64(requiredSpace)), formatSize(int64(availableBytes)))
					errorChan <- errorMsg
					logMessage(errorMsg, true)
					return
				}
                 logMessage(fmt.Sprintf("Disk space check on %s: OK (Available: %s, Required: approx %s)", mp, formatSize(int64(availableBytes)), formatSize(int64(requiredSpace))), debug)
			} else {
                logMessage(fmt.Sprintf("Warning: Could not check disk space on %s: %v. Proceeding anyway.", mp, err), true)
            }

			// 創建測試數據 (在模式循環之外創建一次)
            logMessage(fmt.Sprintf("Generating %s of random data for tests on %s...", formatSize(config.FileSize), mp), debug)
			data := make([]byte, config.FileSize)
			if _, err := rand.Read(data); err != nil {
				errorMsg := fmt.Sprintf("Failed to generate random data for %s: %v", mp, err)
				errorChan <- errorMsg
				logMessage(errorMsg, true)
				return
			}
             logMessage(fmt.Sprintf("Random data generated for %s.", mp), debug)


			// 針對每種測試模式進行測試
			modeWg := &sync.WaitGroup{} // 用於等待同一掛載點下不同模式的測試完成
			for _, mode := range testModes {
				modeWg.Add(1)
				go func(currentMode string) {
					defer modeWg.Done()
					filePath := filepath.Join(mp, fmt.Sprintf("stress_test_%s_%d.dat", currentMode, rand.Intn(10000))) // 加隨機數避免衝突

                    // 在 goroutine 開始時記錄，確保能追蹤到啟動
                    logMessage(fmt.Sprintf("Starting disk test goroutine for mount point: %s (mode: %s, file: %s, size: %s, block: %s)",
							mp, currentMode, filepath.Base(filePath), formatSize(config.FileSize), formatSize(config.BlockSize)), debug)

					// 循環執行讀寫測試直到收到停止信號
					iteration := 0
					for {
                        iteration++
                        logMessage(fmt.Sprintf("Mount %s, Mode %s, Iteration %d: Starting cycle.", mp, currentMode, iteration), debug)

						select {
						case <-stop:
							os.Remove(filePath) // 清理測試文件
							logMessage(fmt.Sprintf("Disk test stopped on %s (mode: %s, file: %s)", mp, currentMode, filepath.Base(filePath)), debug)
							return // 退出此模式的 goroutine
						default:
							// --- 寫入測試 ---
                            logMessage(fmt.Sprintf("Mount %s, Mode %s, Iteration %d: Performing write...", mp, currentMode, iteration), debug)
							writeStart := time.Now()
							writeErr := performDiskWrite(filePath, data, currentMode, config.BlockSize)
                            writeDuration := time.Since(writeStart)

							if writeErr != nil {
								errorMsg := fmt.Sprintf("Disk write error on %s (mode: %s, file: %s, iter: %d, duration: %v): %v", mp, currentMode, filepath.Base(filePath), iteration, writeDuration, writeErr)
								errorChan <- errorMsg
								logMessage(errorMsg, true) // Log error
                                os.Remove(filePath) // 寫入失敗後嘗試清理
								time.Sleep(2 * time.Second) // 發生錯誤時稍等更久
								continue // 繼續下一次循環嘗試
							}

							// 寫入成功，計算速度
							writeSpeedMBps := float64(0)
							if writeDuration.Seconds() > 0 {
								writeSpeedMBps = float64(config.FileSize) / writeDuration.Seconds() / (1024 * 1024)
							}
							logMessage(fmt.Sprintf("Disk write on %s (mode: %s, iter: %d): %.2f MB/s (%s in %v)",
								mp, currentMode, iteration, writeSpeedMBps, formatSize(config.FileSize), writeDuration), debug)

							// --- 讀取和驗證測試 ---
                            logMessage(fmt.Sprintf("Mount %s, Mode %s, Iteration %d: Performing read and verify...", mp, currentMode, iteration), debug)
							readStart := time.Now()
							// 確保讀取和寫入使用相同的模式
							readErr := performDiskReadAndVerify(filePath, data, currentMode, config.BlockSize)
                            readDuration := time.Since(readStart)

							if readErr != nil {
								errorMsg := fmt.Sprintf("Disk read/verify error on %s (mode: %s, file: %s, iter: %d, duration: %v): %v", mp, currentMode, filepath.Base(filePath), iteration, readDuration, readErr)
								errorChan <- errorMsg
								logMessage(errorMsg, true) // Log error
                                os.Remove(filePath) // 讀取/驗證失敗後也嘗試清理
								time.Sleep(2 * time.Second) // 發生錯誤時稍等更久
								continue // 繼續下一次循環嘗試
							}

                            // 讀取和驗證成功，計算速度
							readSpeedMBps := float64(0)
							if readDuration.Seconds() > 0 {
								readSpeedMBps = float64(config.FileSize) / readDuration.Seconds() / (1024 * 1024)
							}
							logMessage(fmt.Sprintf("Disk read/verify on %s (mode: %s, iter: %d): %.2f MB/s (%s in %v)",
								mp, currentMode, iteration, readSpeedMBps, formatSize(config.FileSize), readDuration), debug)

							// --- 更新性能統計 (只記錄最佳性能) ---
							perfStats.mu.Lock()
                            diskPerfKey := fmt.Sprintf("%s|%s|%d", mp, currentMode, config.BlockSize)
                            found := false
                            for i, dp := range perfStats.Disk {
                                // 使用組合鍵來查找匹配的記錄
                                existingKey := fmt.Sprintf("%s|%s|%d", dp.MountPoint, dp.Mode, dp.BlockSize)
                                if existingKey == diskPerfKey {
                                     if readSpeedMBps > dp.ReadSpeed {
                                        perfStats.Disk[i].ReadSpeed = readSpeedMBps
                                        logMessage(fmt.Sprintf("Updated best read speed for %s: %.2f MB/s", diskPerfKey, readSpeedMBps), debug)
                                    }
                                    if writeSpeedMBps > dp.WriteSpeed {
                                        perfStats.Disk[i].WriteSpeed = writeSpeedMBps
                                        logMessage(fmt.Sprintf("Updated best write speed for %s: %.2f MB/s", diskPerfKey, writeSpeedMBps), debug)
                                    }
                                    found = true
                                    break
                                }
                            }
                            if !found {
                                // 如果是第一次記錄這個組合，則添加新條目
                                newPerf := DiskPerformance{
                                    MountPoint: mp,
                                    Mode:       currentMode,
                                    BlockSize:  config.BlockSize,
                                    ReadSpeed:  readSpeedMBps,
                                    WriteSpeed: writeSpeedMBps,
                                }
                                perfStats.Disk = append(perfStats.Disk, newPerf)
                                logMessage(fmt.Sprintf("Added initial perf record for %s: Read=%.2f MB/s, Write=%.2f MB/s", diskPerfKey, readSpeedMBps, writeSpeedMBps), debug)
                            }
							perfStats.mu.Unlock()

							// 可選：每次成功循環後清理文件，避免佔用過多空間（但會增加 I/O 開銷）
							// if err := os.Remove(filePath); err != nil {
                            //     logMessage(fmt.Sprintf("Warning: Failed to remove test file %s after successful cycle: %v", filePath, err), true)
                            // }

							// 短暫延遲避免 CPU/Disk 滿載，並給系統一些喘息時間
                            logMessage(fmt.Sprintf("Mount %s, Mode %s, Iteration %d: Cycle completed successfully. Sleeping.", mp, currentMode, iteration), debug)
							time.Sleep(150 * time.Millisecond) // 稍微增加延遲
						}
					} // end infinite loop for mode
				}(mode) // 傳遞 mode 的當前值
			} // end loop over modes
			modeWg.Wait() // 等待該掛載點的所有模式測試完成 (雖然理論上它們在收到 stop 前不會結束)
            logMessage(fmt.Sprintf("All test modes finished or stopped for mount point %s.", mp), debug)
		}(mountPoint) // 傳遞 mountPoint 的當前值
	} // end loop over mount points

	// 等待所有掛載點的 goroutine 完成 (通常是收到 stop 信號後)
	mountWg.Wait()
    logMessage("All mount point test goroutines have finished.", debug)
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
				go diskTest(&wg, stop, errorChan, diskConfig, perfStats, debug)
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
