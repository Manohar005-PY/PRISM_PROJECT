package com.prism.assistant.camera

import androidx.camera.core.ImageProxy
import com.prism.assistant.bridge.CandidateGeometry
import com.prism.assistant.bridge.PrismObservation
import java.nio.ByteBuffer
import java.util.ArrayDeque
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.tan

/**
 * Structural 2D geometric shape classification of candidate cluster.
 */
enum class CandidateShapeClass {
    COMPACT_VOLUME,
    ELONGATED_STRUCTURE,
    PLANAR_FLOOR_FEATURE,
    UNCERTAIN
}

/**
 * Geometric rejection reason enumeration for diagnostic traceability and telemetry.
 */
enum class GeometryRejectReason {
    INVALID_GROUND_RAY,
    INVALID_DEPTH,
    INVALID_HEIGHT,
    INVALID_WIDTH,
    INVALID_ASPECT_RATIO,
    OUT_OF_LATERAL_RANGE,
    OUT_OF_FORWARD_RANGE,
    GEOMETRIC_UNCERTAINTY_TOO_HIGH,
    PLANAR_FLOOR_FEATURE
}

/**
 * Result of Stage 3 geometric validation for a candidate visual cluster.
 */
data class GeometricValidationResult(
    val valid: Boolean,
    val score: Double,
    val reasons: Set<GeometryRejectReason> = emptySet(),
    val groundX: Double = 0.0,
    val groundZ: Double = 0.0,
    val physicalHeightMeters: Double = 0.0,
    val physicalWidthMeters: Double = 0.0,
    val aspectRatio: Double = 0.0,
    val shapeClass: CandidateShapeClass = CandidateShapeClass.UNCERTAIN,
    val fillRatio: Double = 1.0,
    val floorStructureLikelihood: Double = 0.0,
    val groundPlaneCoherence: Double = 1.0
)

/**
 * Dedicated, decoupled configuration parameters for perception and geometric validation.
 */
data class PerceptionConfig(
    val cameraHeightMeters: Double = 1.30,   // Standard chest / handheld height in meters (1.3m)
    val pitchAngleRadians: Double = 0.20944, // ~12.0 degrees downward tilt in radians
    val minPhysicalHeightM: Double = 0.32,   // Rejects flat floor tile seams, carpet lines, floor shadows (<0.32m)
    val minVolumetricHeightM: Double = 0.22, // Minimum height for verified compact volumetric 3D objects
    val maxPhysicalHeightM: Double = 2.40,   // Plausible upper limit for standing physical obstacles in human spaces
    val minDepthM: Double = 0.40,            // Minimum reliable monocular forward depth
    val maxDepthM: Double = 6.50,            // Maximum reliable monocular indoor forward depth
    val maxLateralExtentM: Double = 1.25,    // Navigable corridor lateral boundary cutoff
    val minCandidateWidthM: Double = 0.10,   // Minimum physical width (rejects single-pixel needle artifacts)
    val maxCandidateWidthM: Double = 2.20,   // Maximum physical width (rejects full-width sweep blurs)
    val minAspectRatio: Double = 0.15,       // Minimum height-to-width ratio
    val maxAspectRatio: Double = 5.50,       // Maximum height-to-width ratio
    val closeHazardDistanceM: Double = 1.00  // Proximity threshold for fast safety confirmation
)

/**
 * Visual candidate cluster extracted from 2D luminance edge grid.
 */
data class CandidateCluster(
    var minCol: Int,
    var maxCol: Int,
    var minRow: Int,
    var maxRow: Int,
    var sumCol: Long,
    var sumRow: Long,
    var size: Int,
    var totalEdgeEnergy: Float = 0f,
    var leftBottomRow: Int = 0,
    var centerBottomRow: Int = 0,
    var rightBottomRow: Int = 0,
    var fillRatio: Double = 1.0,
    var shapeClass: CandidateShapeClass = CandidateShapeClass.UNCERTAIN,
    var floorStructureLikelihood: Double = 0.0,
    var groundPlaneCoherence: Double = 1.0
)

/**
 * 3D Ground-Plane Physical Perception Analyzer with Stage 3 Geometric Validation.
 *
 * Architecture:
 * Candidate Grid Extraction
 *      ↓
 * BFS Connected Components
 *      ↓
 * Pinhole Ground-Plane Ray Projection
 *      ↓
 * Physical Dimensions & Ground-Contact Plausibility
 *      ↓
 * Hard Geometric Sanity Checks + Soft Confidence Scoring
 *      ↓
 * Valid Physical Observation Generation
 */
class AndroidPerceptionAnalyzer(
    val config: PerceptionConfig = PerceptionConfig()
) {

    // Secondary constructor for backward compatibility
    constructor(cameraHeightMeters: Double, pitchAngleRadians: Double) : this(
        PerceptionConfig(
            cameraHeightMeters = cameraHeightMeters,
            pitchAngleRadians = pitchAngleRadians
        )
    )

    val cameraHeightMeters: Double get() = config.cameraHeightMeters
    val pitchAngleRadians: Double get() = config.pitchAngleRadians

    private val gridCols = 32
    private val gridRows = 24

    fun analyzeFrame(
        imageProxy: ImageProxy,
        timestamp: Double,
        orientation: com.prism.assistant.sensor.CameraOrientation? = null
    ): List<PrismObservation> {
        val plane = imageProxy.planes[0]
        val buffer: ByteBuffer = plane.buffer
        val width = imageProxy.width
        val height = imageProxy.height
        val rowStride = plane.rowStride
        val pixelStride = plane.pixelStride

        if (width <= 0 || height <= 0 || buffer.remaining() <= 0) {
            return emptyList()
        }

        // Camera Pinhole Intrinsics Calibration
        val fx = width * 0.95
        val fy = height * 0.95
        val cx = width / 2.0
        val cy = height / 2.0

        // Horizon row calculation: v_horizon = cy - fy * tan(pitch)
        val horizonV = cy - fy * tan(config.pitchAngleRadians)
        val horizonRow = max(1, (horizonV / height * gridRows).toInt())

        val stepX = width / gridCols
        val stepY = height / gridRows

        val grid = Array(gridRows) { FloatArray(gridCols) }

        // 1. Sample luminance grid from Y plane
        for (r in 0 until gridRows) {
            val y = min(height - 1, r * stepY + stepY / 2)
            for (c in 0 until gridCols) {
                val x = min(width - 1, c * stepX + stepX / 2)
                val offset = y * rowStride + x * pixelStride
                if (offset < buffer.limit()) {
                    val lum = buffer.get(offset).toInt() and 0xFF
                    grid[r][c] = lum.toFloat()
                }
            }
        }

        // 2. Compute local spatial edge energy below horizon
        val edgeEnergy = Array(gridRows) { FloatArray(gridCols) }
        val isObstacleCell = Array(gridRows) { BooleanArray(gridCols) }
        val startRow = max(horizonRow + 1, (gridRows * 0.30).toInt())

        for (r in startRow until (gridRows - 1)) {
            for (c in 1 until (gridCols - 1)) {
                val gradX = abs(grid[r][c + 1] - grid[r][c - 1])
                val gradY = abs(grid[r + 1][c] - grid[r - 1][c])
                val energy = gradX + gradY
                edgeEnergy[r][c] = energy

                // Local 3x3 contrast against neighborhood
                var localMean = 0f
                for (dr in -1..1) {
                    for (dc in -1..1) {
                        localMean += grid[r + dr][c + dc]
                    }
                }
                localMean /= 9f
                val localContrast = abs(grid[r][c] - localMean)

                // Requires prominent visual edge structure (> 38.0) and contrast (> 24.0)
                if (energy > 38f && localContrast > 24f) {
                    isObstacleCell[r][c] = true
                }
            }
        }

        // 3. Multi-Cluster Connected Component Labeling via BFS
        val visited = Array(gridRows) { BooleanArray(gridCols) }
        val clusters = mutableListOf<CandidateCluster>()

        val dRow = intArrayOf(-1, 1, 0, 0)
        val dCol = intArrayOf(0, 0, -1, 1)

        for (r in startRow until (gridRows - 1)) {
            for (c in 1 until (gridCols - 1)) {
                if (isObstacleCell[r][c] && !visited[r][c]) {
                    val cluster = CandidateCluster(
                        minCol = c, maxCol = c,
                        minRow = r, maxRow = r,
                        sumCol = 0, sumRow = 0,
                        size = 0,
                        totalEdgeEnergy = 0f
                    )
                    val queue = ArrayDeque<Pair<Int, Int>>()
                    queue.add(Pair(r, c))
                    visited[r][c] = true

                    val colBottomRows = mutableMapOf<Int, Int>()

                    while (queue.isNotEmpty()) {
                        val (currR, currC) = queue.removeFirst()
                        cluster.minCol = min(cluster.minCol, currC)
                        cluster.maxCol = max(cluster.maxCol, currC)
                        cluster.minRow = min(cluster.minRow, currR)
                        cluster.maxRow = max(cluster.maxRow, currR)
                        cluster.sumCol += currC
                        cluster.sumRow += currR
                        cluster.totalEdgeEnergy += edgeEnergy[currR][currC]
                        cluster.size++

                        val prevBot = colBottomRows[currC] ?: -1
                        if (currR > prevBot) {
                            colBottomRows[currC] = currR
                        }

                        for (i in 0..3) {
                            val nr = currR + dRow[i]
                            val nc = currC + dCol[i]
                            if (nr in startRow until (gridRows - 1) && nc in 1 until (gridCols - 1)) {
                                if (isObstacleCell[nr][nc] && !visited[nr][nc]) {
                                    visited[nr][nc] = true
                                    queue.add(Pair(nr, nc))
                                }
                            }
                        }
                    }

                    if (cluster.size >= 6) {
                        // Stage 2 Candidate Morphology Descriptors
                        val spanRows = cluster.maxRow - cluster.minRow + 1
                        val spanCols = cluster.maxCol - cluster.minCol + 1
                        val fillRatio = (cluster.size.toDouble() / max(1, spanRows * spanCols)).coerceIn(0.0, 1.0)
                        val aspect2D = spanRows.toDouble() / max(1, spanCols)

                        val centerC = (cluster.sumCol / max(1, cluster.size)).toInt()
                        val leftBot = colBottomRows[cluster.minCol] ?: cluster.maxRow
                        val centerBot = colBottomRows[centerC] ?: cluster.maxRow
                        val rightBot = colBottomRows[cluster.maxCol] ?: cluster.maxRow

                        val groundRowDiff = max(abs(leftBot - centerBot), abs(rightBot - centerBot))
                        val groundCoherence = max(0.10, 1.0 - (groundRowDiff / 3.0)).coerceIn(0.10, 1.0)

                        // Planar Floor Structure Likelihood Scoring
                        val floorLikelihood = when {
                            spanRows <= 1 -> 0.95
                            spanRows == 2 && spanCols >= 4 -> 0.85
                            spanRows == 2 -> 0.65
                            aspect2D <= 0.25 -> 0.75
                            aspect2D <= 0.40 && fillRatio < 0.35 -> 0.60
                            spanRows >= 4 && fillRatio >= 0.40 -> 0.10
                            else -> 0.30
                        }

                        val shapeClass = when {
                            floorLikelihood >= 0.65 -> CandidateShapeClass.PLANAR_FLOOR_FEATURE
                            aspect2D > 3.0 || (aspect2D < 0.25 && fillRatio < 0.40) -> CandidateShapeClass.ELONGATED_STRUCTURE
                            spanRows >= 3 && fillRatio >= 0.40 && floorLikelihood < 0.35 -> CandidateShapeClass.COMPACT_VOLUME
                            else -> CandidateShapeClass.UNCERTAIN
                        }

                        cluster.leftBottomRow = leftBot
                        cluster.centerBottomRow = centerBot
                        cluster.rightBottomRow = rightBot
                        cluster.fillRatio = fillRatio
                        cluster.floorStructureLikelihood = floorLikelihood
                        cluster.shapeClass = shapeClass
                        cluster.groundPlaneCoherence = groundCoherence

                        clusters.add(cluster)
                    }
                }
            }
        }

        // 4. Stage 3: Explicit Geometric Validation & Observation Extraction
        val observations = mutableListOf<PrismObservation>()

        val effPitch = orientation?.pitchRadians ?: config.pitchAngleRadians
        val effRoll = orientation?.rollRadians ?: 0.0

        for (cluster in clusters) {
            val geomResult = validateCandidateGeometry(
                cluster = cluster,
                fx = fx,
                fy = fy,
                cx = cx,
                cy = cy,
                stepX = stepX,
                stepY = stepY,
                pitchRadians = effPitch,
                rollRadians = effRoll
            )

            // Hard rejection filter: Drop clearly implausible physical structures
            if (!geomResult.valid) {
                continue
            }

            val z = geomResult.groundZ
            val x = geomResult.groundX
            val estimatedPhysicalHeight = geomResult.physicalHeightMeters
            val widthMeters = geomResult.physicalWidthMeters
            val aspectRatio = geomResult.aspectRatio
            val clusterHeightRows = (cluster.maxRow - cluster.minRow + 1)

            // Semantic Classification based on validated 3D geometry
            val className = when {
                estimatedPhysicalHeight >= 1.20 || (aspectRatio >= 1.3 && clusterHeightRows >= 6) -> "person"
                estimatedPhysicalHeight in 0.22..0.65 -> "low_obstacle"
                else -> "obstacle"
            }

            val covariance = 0.04 + (0.015 * z)
            val confidence = geomResult.score

            val candGeom = CandidateGeometry(
                widthMeters = widthMeters,
                heightMeters = estimatedPhysicalHeight,
                aspectRatio = aspectRatio,
                groundContactV = (cluster.maxRow.toDouble() + 0.5) * stepY,
                topV = cluster.minRow.toDouble() * stepY,
                leftCol = cluster.minCol,
                rightCol = cluster.maxCol,
                topRow = cluster.minRow,
                bottomRow = cluster.maxRow,
                gradientEnergy = (cluster.totalEdgeEnergy / max(1, cluster.size)).toDouble(),
                localContrast = 0.0,
                clusterSize = cluster.size,
                geometryValid = true,
                geometryScore = geomResult.score,
                rejectReason = null,
                shapeClass = geomResult.shapeClass.name,
                fillRatio = geomResult.fillRatio,
                floorStructureLikelihood = geomResult.floorStructureLikelihood,
                groundPlaneCoherence = geomResult.groundPlaneCoherence
            )

            observations.add(
                PrismObservation(
                    x = x,
                    z = z,
                    covariance = covariance,
                    confidence = confidence,
                    className = className,
                    geometry = candGeom,
                    geometricConfidence = geomResult.score
                )
            )
        }

        return observations
    }

    /**
     * Explicit Stage 3 Geometric Validation function.
     * Evaluates pinhole ground projection with dynamic pitch and roll compensation,
     * physical height, width, aspect ratio, shape morphology, multi-point ground coherence.
     */
    fun validateCandidateGeometry(
        cluster: CandidateCluster,
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        stepX: Int,
        stepY: Int,
        pitchRadians: Double = config.pitchAngleRadians,
        rollRadians: Double = 0.0
    ): GeometricValidationResult {
        val cosTheta = cos(pitchRadians)
        val sinTheta = sin(pitchRadians)
        val cosPhi = cos(rollRadians)
        val sinPhi = sin(rollRadians)

        val centerCol = cluster.sumCol.toDouble() / max(1, cluster.size)
        val bottomRow = cluster.maxRow.toDouble()
        val topRow = cluster.minRow.toDouble()

        // Full frame pixel coordinates
        val pixelU = (centerCol + 0.5) * stepX
        val pixelVBottom = (bottomRow + 0.5) * stepY
        val pixelVTop = topRow * stepY

        val xNorm = (pixelU - cx) / fx
        val yNormBottom = (pixelVBottom - cy) / fy

        // 3D Optical Ray Rotation (Roll phi around Z-axis, then Pitch theta around X-axis)
        val rxBottom = xNorm * cosPhi - yNormBottom * sinPhi
        val ryBottom = (xNorm * sinPhi + yNormBottom * cosPhi) * cosTheta + sinTheta
        val rzBottom = -(xNorm * sinPhi + yNormBottom * cosPhi) * sinTheta + cosTheta

        val denom = ryBottom
        val reasons = mutableSetOf<GeometryRejectReason>()

        // 1. Ray Ground Intersection Check
        if (denom <= 0.04 || denom.isNaN() || denom.isInfinite()) {
            reasons.add(GeometryRejectReason.INVALID_GROUND_RAY)
            return GeometricValidationResult(
                valid = false,
                score = 0.0,
                reasons = reasons,
                shapeClass = cluster.shapeClass,
                fillRatio = cluster.fillRatio,
                floorStructureLikelihood = cluster.floorStructureLikelihood,
                groundPlaneCoherence = cluster.groundPlaneCoherence
            )
        }

        val t = config.cameraHeightMeters / denom
        val z = t * rzBottom
        val x = t * rxBottom

        // 2. Forward Depth Validity Check
        if (z.isNaN() || z.isInfinite() || z <= 0.0) {
            reasons.add(GeometryRejectReason.INVALID_DEPTH)
        } else if (z < config.minDepthM || z > config.maxDepthM) {
            reasons.add(GeometryRejectReason.OUT_OF_FORWARD_RANGE)
        }

        // 3. Lateral Corridor Range Check
        if (x.isNaN() || x.isInfinite() || abs(x) > config.maxLateralExtentM) {
            reasons.add(GeometryRejectReason.OUT_OF_LATERAL_RANGE)
        }

        // 4. 3D Physical Standing Height Check & Volumetric Distinction
        val yNormTop = (pixelVTop - cy) / fy
        val yNormTopRot = xNorm * sinPhi + yNormTop * cosPhi
        val topDenom = -yNormTopRot * sinTheta + cosTheta
        val estimatedPhysicalHeight = if (topDenom > 0.05) {
            config.cameraHeightMeters - (z * (yNormTopRot * cosTheta + sinTheta) / topDenom)
        } else {
            1.5 // Extends to/above horizon -> tall upright structure
        }

        val isCompactVolume = (cluster.shapeClass == CandidateShapeClass.COMPACT_VOLUME && cluster.floorStructureLikelihood < 0.35)
        val effectiveMinHeight = if (isCompactVolume) config.minVolumetricHeightM else config.minPhysicalHeightM

        if (estimatedPhysicalHeight.isNaN() || estimatedPhysicalHeight.isInfinite()) {
            reasons.add(GeometryRejectReason.INVALID_HEIGHT)
        } else if (estimatedPhysicalHeight < effectiveMinHeight || estimatedPhysicalHeight > config.maxPhysicalHeightM) {
            reasons.add(GeometryRejectReason.INVALID_HEIGHT)
        } else if (cluster.shapeClass == CandidateShapeClass.PLANAR_FLOOR_FEATURE && estimatedPhysicalHeight < 0.42) {
            reasons.add(GeometryRejectReason.PLANAR_FLOOR_FEATURE)
        }

        // 5. Physical Width & Aspect Ratio Checks
        val clusterWidthCols = (cluster.maxCol - cluster.minCol + 1)
        val widthMeters = (clusterWidthCols.toDouble() / gridCols) * (z * 0.95)

        if (widthMeters.isNaN() || widthMeters.isInfinite() || widthMeters < config.minCandidateWidthM || widthMeters > config.maxCandidateWidthM) {
            reasons.add(GeometryRejectReason.INVALID_WIDTH)
        }

        val aspectRatio = estimatedPhysicalHeight / max(0.01, widthMeters)
        if (aspectRatio.isNaN() || aspectRatio.isInfinite() || aspectRatio < config.minAspectRatio || aspectRatio > config.maxAspectRatio) {
            reasons.add(GeometryRejectReason.INVALID_ASPECT_RATIO)
        }

        // If any hard rejection triggered -> Return invalid
        if (reasons.isNotEmpty()) {
            return GeometricValidationResult(
                valid = false,
                score = 0.0,
                reasons = reasons,
                groundX = if (x.isNaN()) 0.0 else x,
                groundZ = if (z.isNaN()) 0.0 else z,
                physicalHeightMeters = if (estimatedPhysicalHeight.isNaN()) 0.0 else estimatedPhysicalHeight,
                physicalWidthMeters = if (widthMeters.isNaN()) 0.0 else widthMeters,
                aspectRatio = if (aspectRatio.isNaN()) 0.0 else aspectRatio,
                shapeClass = cluster.shapeClass,
                fillRatio = cluster.fillRatio,
                floorStructureLikelihood = cluster.floorStructureLikelihood,
                groundPlaneCoherence = cluster.groundPlaneCoherence
            )
        }

        // 6. Normalized Geometric Confidence Scoring (0.0 to 1.0)
        var confidence = min(0.98, 0.70 + (cluster.size / 25.0))

        // Morphology & Shape Scoring
        if (cluster.shapeClass == CandidateShapeClass.COMPACT_VOLUME) {
            confidence += 0.08
        } else if (cluster.shapeClass == CandidateShapeClass.PLANAR_FLOOR_FEATURE) {
            confidence -= 0.25
        } else if (cluster.shapeClass == CandidateShapeClass.ELONGATED_STRUCTURE) {
            confidence -= 0.12
        }

        // Planar floor likelihood penalty
        confidence -= 0.20 * cluster.floorStructureLikelihood

        // Ground plane coherence
        confidence += 0.10 * (cluster.groundPlaneCoherence - 0.50)

        // Distance uncertainty soft penalty
        if (z > 4.5) {
            confidence -= 0.04 * (z - 4.5)
        }
        // Borderline height soft penalty
        if (estimatedPhysicalHeight in effectiveMinHeight..(effectiveMinHeight + 0.08)) {
            confidence -= 0.08
        }
        // Aspect ratio soft penalty
        if (aspectRatio < 0.25 || aspectRatio > 4.5) {
            confidence -= 0.08
        }

        val finalScore = max(0.10, min(0.99, confidence))

        return GeometricValidationResult(
            valid = true,
            score = finalScore,
            reasons = emptySet(),
            groundX = x,
            groundZ = z,
            physicalHeightMeters = estimatedPhysicalHeight,
            physicalWidthMeters = widthMeters,
            aspectRatio = aspectRatio,
            shapeClass = cluster.shapeClass,
            fillRatio = cluster.fillRatio,
            floorStructureLikelihood = cluster.floorStructureLikelihood,
            groundPlaneCoherence = cluster.groundPlaneCoherence
        )
    }
}
