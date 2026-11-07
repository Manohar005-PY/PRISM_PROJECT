package com.prism.assistant

import com.prism.assistant.bridge.CandidateGeometry
import com.prism.assistant.bridge.DeterministicPrismBridge
import com.prism.assistant.bridge.PrismObservation
import com.prism.assistant.bridge.TrackStatus
import com.prism.assistant.accessibility.AccessibilityController
import com.prism.assistant.accessibility.AudioPriority
import com.prism.assistant.accessibility.NavigationState
import com.prism.assistant.accessibility.PrismBaselineConfig
import com.prism.assistant.accessibility.ReleaseMode
import com.prism.assistant.accessibility.SafeDegradationState
import com.prism.assistant.camera.AndroidPerceptionAnalyzer
import com.prism.assistant.camera.CandidateCluster
import com.prism.assistant.camera.CandidateShapeClass
import com.prism.assistant.camera.FrameBackpressurePolicy
import com.prism.assistant.camera.GeometryRejectReason
import com.prism.assistant.camera.PerceptionConfig
import com.prism.assistant.sensor.OrientationSourceState
import com.prism.assistant.sensor.StepDetectorConfig
import com.prism.assistant.sensor.UserMotionEstimate
import com.prism.assistant.sensor.UserMotionEstimator
import com.prism.assistant.sensor.UserMotionInferenceMode
import com.prism.assistant.spatial.MultisensoryMode
import com.prism.assistant.spatial.SpatialCue
import com.prism.assistant.spatial.SpatialCueType
import com.prism.assistant.warning.DistanceBand
import com.prism.assistant.warning.HazardLifecycleState
import com.prism.assistant.warning.MovementState
import com.prism.assistant.warning.SpatialDirection
import com.prism.assistant.warning.SpatialMappingConfig
import com.prism.assistant.warning.UserMotionState
import com.prism.assistant.warning.WarningContext
import com.prism.assistant.warning.WarningMessageGenerator
import com.prism.assistant.warning.WarningSeverity
import kotlin.math.abs
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class PrismBridgeUnitTest {

    private lateinit var bridge: DeterministicPrismBridge
    private lateinit var backpressure: FrameBackpressurePolicy

    @Before
    fun setUp() {
        bridge = DeterministicPrismBridge()
        bridge.initialize()
        backpressure = FrameBackpressurePolicy(maxQueueDepth = 1, maxStaleDeltaSeconds = 0.25)
    }

    @Test
    fun testEmptyObservationsYieldsContinueAction() {
        val result = bridge.processFrame(timestamp = 0.0, observations = emptyList())
        assertEquals("continue", result.decision.action)
        assertEquals("safe", result.decision.riskLevel)
        assertEquals(0, result.activeTracks)
        assertEquals(1, result.cues.size)
        assertEquals("visual", result.cues[0].channel)
    }

    @Test
    fun testCloseObstacleYieldsEmergencyStop() {
        val obs = listOf(PrismObservation(x = 0.0, z = 0.8, className = "person"))
        val result = bridge.processFrame(timestamp = 1.0, observations = obs)
        assertEquals("emergency_stop", result.decision.action)
        assertEquals("critical", result.decision.riskLevel)
        assertTrue(result.cues.any { it.channel == "speech" && (it.text == "Stop" || it.text == "Stop. Obstacle directly ahead.") })
        assertTrue(result.cues.any { it.channel == "haptic" && it.hapticPattern == "emergency" })
        assertTrue(result.cues.any { it.channel == "spatial_audio" && it.direction == 0.0 })
    }

    @Test
    fun testMediumDistanceObstacleYieldsTurnEvasion() {
        val obs = listOf(PrismObservation(x = 0.3, z = 2.0, className = "obstacle"))
        val result = bridge.processFrame(timestamp = 2.0, observations = obs)
        assertEquals("turn_left", result.decision.action)
        assertEquals("high", result.decision.riskLevel)
        assertNotNull(result.decision.recommendedHeading)
        assertTrue(result.decision.recommendedHeading!! < 0.0)
    }

    @Test
    fun testMultiEntitySimultaneousTracking() {
        // Feed two distinct simultaneous entities
        val obs = listOf(
            PrismObservation(x = 0.3, z = 2.0, className = "person"),
            PrismObservation(x = -1.4, z = 3.8, className = "obstacle")
        )
        val result = bridge.processFrame(timestamp = 3.0, observations = obs)
        assertEquals(2, result.activeTracks)
        assertEquals(2, result.trackList.size)
        // Primary threat should be the closer obstacle (at z=2.0)
        assertNotNull(result.decision.primaryThreatId)
        assertEquals("high", result.decision.riskLevel)
    }

    @Test
    fun testBackpressureAndStaleDetection() {
        backpressure.onFrameCaptured(0.0)
        assertFalse(backpressure.isStale(frameTimestamp = 0.0, currentTimestamp = 0.1))
        assertTrue(backpressure.isStale(frameTimestamp = 0.0, currentTimestamp = 0.4))
    }

    @Test
    fun testSideObstacleYieldsSafeContinue() {
        // Obstacle situated far to the side (|x| = 1.35m) outside direct path
        val obs = listOf(PrismObservation(x = 1.35, z = 3.0, className = "obstacle"))
        val result = bridge.processFrame(timestamp = 4.0, observations = obs)
        assertEquals("continue", result.decision.action)
        assertEquals("safe", result.decision.riskLevel)
        assertTrue(result.cues.none { it.channel == "speech" })
    }

    @Test
    fun testTentativeTrackNoiseFiltering() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        // Frame 1: Noise blip at medium distance (z = 3.5m)
        val frame1Obs = listOf(PrismObservation(x = 0.1, z = 3.5, className = "obstacle"))
        val res1 = multiHitBridge.processFrame(timestamp = 5.0, observations = frame1Obs)
        assertEquals(1, res1.activeTracks)

        // Frame 2: Noise blip disappeared (empty frame) -> track should be dropped immediately without lingering
        val res2 = multiHitBridge.processFrame(timestamp = 5.033, observations = emptyList())
        assertEquals(0, res2.activeTracks)
        assertEquals("continue", res2.decision.action)
        assertEquals("safe", res2.decision.riskLevel)
    }

    @Test
    fun testCandidateGeometryAndTelemetrySerialization() {
        val geom = com.prism.assistant.bridge.CandidateGeometry(
            widthMeters = 0.55,
            heightMeters = 1.45,
            aspectRatio = 1.8,
            groundContactV = 380.0,
            topV = 160.0,
            leftCol = 10,
            rightCol = 18,
            topRow = 8,
            bottomRow = 19,
            gradientEnergy = 48.5,
            localContrast = 28.0,
            clusterSize = 16
        )
        val obs = listOf(
            PrismObservation(
                x = 0.2,
                z = 2.4,
                className = "person",
                geometry = geom,
                geometricConfidence = 0.92
            )
        )
        val res = bridge.processFrame(timestamp = 6.0, observations = obs)
        assertEquals(1, res.activeTracks)
        val trk = res.trackList[0]
        assertEquals(0.55, trk.widthMeters, 0.01)
        assertEquals(1.45, trk.heightMeters, 0.01)
        assertEquals(1.8, trk.aspectRatio, 0.01)
        assertEquals(16, trk.clusterSize)
        assertEquals(0.92, trk.geometricConfidence, 0.01)
        assertEquals("CONFIRMED", trk.status)
    }

    @Test
    fun testValidObstacleGeometryPassesValidation() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val cluster = CandidateCluster(
            minCol = 14,
            maxCol = 18,
            minRow = 10,
            maxRow = 20,
            sumCol = 16L * 15,
            sumRow = 15L * 15,
            size = 15,
            totalEdgeEnergy = 600f
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = cluster,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertTrue("Valid standing obstacle cluster must pass geometric validation", result.valid)
        assertTrue("Reject reasons must be empty for valid geometry", result.reasons.isEmpty())
        assertTrue("Valid obstacle height should be ~1.0m", result.physicalHeightMeters in 0.8..1.3)
        assertTrue("Valid obstacle depth should be ~2.0m", result.groundZ in 1.8..2.3)
        assertTrue("Confidence score should be > 0.70", result.score >= 0.70)
    }

    @Test
    fun testFlatFloorFeatureRejected() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        // Flat floor seam/tile occupying only row 20
        val cluster = CandidateCluster(
            minCol = 12,
            maxCol = 20,
            minRow = 20,
            maxRow = 20,
            sumCol = 16L * 9,
            sumRow = 20L * 9,
            size = 9,
            totalEdgeEnergy = 350f
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = cluster,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertFalse("Flat floor seam (<0.32m height) must be rejected", result.valid)
        assertTrue("Should contain INVALID_HEIGHT reject reason", result.reasons.contains(GeometryRejectReason.INVALID_HEIGHT))
        assertTrue("Physical height must be < 0.32m", result.physicalHeightMeters < 0.32)
    }

    @Test
    fun testInvalidRayAndDepthRejected() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        // Cluster located above the horizon (row 0..2) where ground projection ray does not intersect floor
        val clusterAboveHorizon = CandidateCluster(
            minCol = 14,
            maxCol = 18,
            minRow = 0,
            maxRow = 2,
            sumCol = 16L * 10,
            sumRow = 1L * 10,
            size = 10,
            totalEdgeEnergy = 400f
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = clusterAboveHorizon,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertFalse("Cluster above horizon must be rejected", result.valid)
        assertTrue(
            "Should reject due to INVALID_GROUND_RAY or OUT_OF_FORWARD_RANGE",
            result.reasons.contains(GeometryRejectReason.INVALID_GROUND_RAY) ||
            result.reasons.contains(GeometryRejectReason.OUT_OF_FORWARD_RANGE) ||
            result.reasons.contains(GeometryRejectReason.INVALID_DEPTH)
        )
    }

    @Test
    fun testExtremeGeometryRejected() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        // Full-screen horizontal streak spanning all columns 0..31 (excessive width > 2.5m)
        val clusterWide = CandidateCluster(
            minCol = 0,
            maxCol = 31,
            minRow = 15,
            maxRow = 16,
            sumCol = 16L * 32,
            sumRow = 15L * 32,
            size = 32,
            totalEdgeEnergy = 1200f
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = clusterWide,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertFalse("Excessively wide candidate must be rejected", result.valid)
        assertTrue(
            "Should contain INVALID_WIDTH, INVALID_ASPECT_RATIO, or INVALID_HEIGHT",
            result.reasons.contains(GeometryRejectReason.INVALID_WIDTH) ||
            result.reasons.contains(GeometryRejectReason.INVALID_ASPECT_RATIO) ||
            result.reasons.contains(GeometryRejectReason.INVALID_HEIGHT)
        )
    }

    @Test
    fun testGeometricallyValidCloseHazardGetsFastConfirmation() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val validGeom = CandidateGeometry(
            widthMeters = 0.40,
            heightMeters = 0.85,
            aspectRatio = 2.1,
            geometryValid = true,
            geometryScore = 0.90
        )
        val obs = listOf(
            PrismObservation(
                x = 0.0,
                z = 0.8,
                className = "obstacle",
                geometry = validGeom,
                geometricConfidence = 0.90
            )
        )
        val result = multiHitBridge.processFrame(timestamp = 7.0, observations = obs)
        assertEquals(1, result.activeTracks)
        val trk = result.trackList[0]
        assertEquals("Geometrically valid close hazard (<1.0m) must be confirmed immediately", "CONFIRMED", trk.status)
        assertEquals("emergency_stop", result.decision.action)
    }

    @Test
    fun testGeometricallyInvalidCloseCandidateDoesNotGetFastConfirmation() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        // Low confidence / flat feature artifact close to camera
        val invalidGeom = CandidateGeometry(
            widthMeters = 0.40,
            heightMeters = 0.15,
            aspectRatio = 0.35,
            geometryValid = false,
            geometryScore = 0.20,
            rejectReason = "INVALID_HEIGHT"
        )
        val obs = listOf(
            PrismObservation(
                x = 0.0,
                z = 0.8,
                className = "obstacle",
                geometry = invalidGeom,
                geometricConfidence = 0.20
            )
        )
        val result = multiHitBridge.processFrame(timestamp = 8.0, observations = obs)
        assertEquals(1, result.activeTracks)
        val trk = result.trackList[0]
        assertEquals("Geometrically invalid close candidate must NOT get fast confirmation", "TENTATIVE", trk.status)
    }

    @Test
    fun testStablePhysicalObstacleConfidenceAccumulates() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.45,
            heightMeters = 1.10,
            aspectRatio = 2.44,
            groundContactV = 350.0,
            geometryValid = true,
            geometryScore = 0.90
        )

        var lastResult: com.prism.assistant.bridge.PrismFrameResult? = null
        for (i in 0 until 5) {
            val t = 1.0 + i * 0.033
            val obs = listOf(
                PrismObservation(
                    x = 0.15,
                    z = 2.50,
                    className = "obstacle",
                    geometry = geom,
                    geometricConfidence = 0.90
                )
            )
            lastResult = multiHitBridge.processFrame(timestamp = t, observations = obs)
        }

        assertNotNull(lastResult)
        assertEquals(1, lastResult!!.activeTracks)
        val trk = lastResult.trackList[0]
        assertEquals("CONFIRMED", trk.status)
        assertTrue("Position consistency should be high for stationary obstacle", trk.positionConsistency >= 0.90)
        assertTrue("Height consistency should be 1.0 for unchanging height", trk.heightConsistency >= 0.95)
        assertTrue("Temporal confidence should accumulate to >0.80", trk.temporalConfidence >= 0.80)
        assertTrue("Physical confidence should be high (>0.80)", trk.physicalConfidence >= 0.80)
    }

    @Test
    fun testMovingPersonApproachingMaintainsHighConfidence() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.50,
            heightMeters = 1.65,
            aspectRatio = 3.30,
            groundContactV = 320.0,
            geometryValid = true,
            geometryScore = 0.95
        )

        var lastResult: com.prism.assistant.bridge.PrismFrameResult? = null
        // Person walking towards camera at ~1.5 m/s
        for (i in 0 until 6) {
            val t = 10.0 + i * 0.05
            val currentZ = 3.50 - (i * 0.075) // 1.5 m/s * 0.05s = 0.075m per step
            val obs = listOf(
                PrismObservation(
                    x = 0.0,
                    z = currentZ,
                    className = "person",
                    geometry = geom,
                    geometricConfidence = 0.95
                )
            )
            lastResult = multiHitBridge.processFrame(timestamp = t, observations = obs)
        }

        assertNotNull(lastResult)
        val trk = lastResult!!.trackList[0]
        assertEquals("CONFIRMED", trk.status)
        assertTrue("Moving person should maintain high depth consistency", trk.depthConsistency >= 0.80)
        assertTrue("Moving person should maintain high temporal confidence", trk.temporalConfidence >= 0.75)
        assertTrue("Approaching velocity should be negative", trk.vz < -0.5)
    }

    @Test
    fun testTemporaryOcclusionDecayAndRecovery() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.40,
            heightMeters = 0.90,
            aspectRatio = 2.25,
            geometryValid = true,
            geometryScore = 0.90
        )

        // Frames 1..3: Object present and confirmed
        for (i in 0 until 3) {
            val t = 20.0 + i * 0.033
            val obs = listOf(PrismObservation(x = 0.2, z = 2.2, className = "obstacle", geometry = geom, geometricConfidence = 0.90))
            multiHitBridge.processFrame(timestamp = t, observations = obs)
        }

        // Frames 4..5: Temporary occlusion (empty frame)
        multiHitBridge.processFrame(timestamp = 20.100, observations = emptyList())
        val resMiss2 = multiHitBridge.processFrame(timestamp = 20.133, observations = emptyList())

        assertEquals(1, resMiss2.activeTracks)
        val lostTrk = resMiss2.trackList[0]
        assertEquals("LOST", lostTrk.status)
        assertTrue("Temporal confidence should decay during misses", lostTrk.temporalConfidence < 0.70)
        assertEquals("MISSED_FRAME_DECAY", lostTrk.confidenceUpdateReason)

        // Frame 6: Reacquired matching observation
        val resReacquired = multiHitBridge.processFrame(
            timestamp = 20.166,
            observations = listOf(PrismObservation(x = 0.2, z = 2.2, className = "obstacle", geometry = geom, geometricConfidence = 0.90))
        )
        assertEquals(1, resReacquired.activeTracks)
        val recoveredTrk = resReacquired.trackList[0]
        assertEquals("CONFIRMED", recoveredTrk.status)
        assertEquals("REACQUIRED_CONFIRMED", recoveredTrk.confidenceUpdateReason)
    }

    @Test
    fun testErraticVisualArtifactFailsTemporalValidation() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        // Feed chaotic erratic measurements (simulating fluctuating floor reflection/glare)
        val erraticDepths = doubleArrayOf(2.0, 0.7, 3.2, 1.1, 2.8)
        val erraticHeights = doubleArrayOf(0.45, 1.40, 0.35, 1.80, 0.50)

        var lastResult: com.prism.assistant.bridge.PrismFrameResult? = null
        for (i in erraticDepths.indices) {
            val t = 30.0 + i * 0.033
            val geom = CandidateGeometry(
                widthMeters = 0.40,
                heightMeters = erraticHeights[i],
                aspectRatio = erraticHeights[i] / 0.40,
                geometryValid = true,
                geometryScore = 0.65
            )
            val obs = listOf(
                PrismObservation(
                    x = 0.0,
                    z = erraticDepths[i],
                    className = "obstacle",
                    geometry = geom,
                    geometricConfidence = 0.65
                )
            )
            lastResult = multiHitBridge.processFrame(timestamp = t, observations = obs)
        }

        assertNotNull(lastResult)
        val trk = lastResult!!.trackList.firstOrNull()
        if (trk != null) {
            assertTrue("Erratic height jumps should penalize height consistency (<0.50)", trk.heightConsistency < 0.50)
            assertTrue("Erratic jumps should keep temporal confidence low (<0.60)", trk.temporalConfidence < 0.60)
        }
    }

    @Test
    fun testStableStaticObstacleUnderCameraMotionPasses() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.45,
            heightMeters = 0.95,
            aspectRatio = 2.11,
            groundContactV = 340.0,
            geometryValid = true,
            geometryScore = 0.90
        )

        var lastResult: com.prism.assistant.bridge.PrismFrameResult? = null
        // User walking forward at 1.0 m/s towards static obstacle
        for (i in 0 until 5) {
            val t = 40.0 + i * 0.033
            val z = 3.00 - (i * 0.033) // 1.0 m/s forward translation
            val obs = listOf(
                PrismObservation(
                    x = 0.10,
                    z = z,
                    className = "obstacle",
                    geometry = geom,
                    geometricConfidence = 0.90
                )
            )
            lastResult = multiHitBridge.processFrame(timestamp = t, observations = obs)
        }

        assertNotNull(lastResult)
        val trk = lastResult!!.trackList[0]
        assertEquals("CONFIRMED", trk.status)
        assertTrue("Coherent motion must not be falsely rejected", trk.temporalConfidence >= 0.70)
        assertTrue("Physical confidence should remain strong", trk.physicalConfidence >= 0.75)
    }

    @Test
    fun testConfidenceBoundariesAndNaNProtection() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.40,
            heightMeters = 1.0,
            aspectRatio = 2.5,
            geometryValid = true,
            geometryScore = 0.85
        )

        val obs = listOf(
            PrismObservation(
                x = Double.NaN,
                z = Double.POSITIVE_INFINITY,
                className = "obstacle",
                geometry = geom,
                geometricConfidence = Double.NaN
            )
        )
        val result = multiHitBridge.processFrame(timestamp = 50.0, observations = obs)
        val trk = result.trackList.firstOrNull()
        if (trk != null) {
            assertFalse("Temporal confidence must not be NaN", trk.temporalConfidence.isNaN())
            assertFalse("Physical confidence must not be NaN", trk.physicalConfidence.isNaN())
            assertTrue("Temporal confidence must be in 0..1", trk.temporalConfidence in 0.0..1.0)
            assertTrue("Physical confidence must be in 0..1", trk.physicalConfidence in 0.0..1.0)
        }
    }

    @Test
    fun testTrackLifecycleStateTransitions() {
        val multiHitBridge = DeterministicPrismBridge(confirmationHits = 2)
        multiHitBridge.initialize()

        val geom = CandidateGeometry(
            widthMeters = 0.40,
            heightMeters = 0.85,
            aspectRatio = 2.12,
            geometryValid = true,
            geometryScore = 0.85
        )

        // 1. TENTATIVE on hit 1 (medium range z=2.5m)
        val res1 = multiHitBridge.processFrame(
            timestamp = 60.0,
            observations = listOf(PrismObservation(x = 0.0, z = 2.5, className = "obstacle", geometry = geom, geometricConfidence = 0.85))
        )
        assertEquals(TrackStatus.TENTATIVE.name, res1.trackList[0].status)

        // 2. CONFIRMED on hit 2
        val res2 = multiHitBridge.processFrame(
            timestamp = 60.033,
            observations = listOf(PrismObservation(x = 0.0, z = 2.5, className = "obstacle", geometry = geom, geometricConfidence = 0.85))
        )
        assertEquals(TrackStatus.CONFIRMED.name, res2.trackList[0].status)

        // 3. LOST on miss 1
        val res3 = multiHitBridge.processFrame(timestamp = 60.066, observations = emptyList())
        assertEquals(TrackStatus.LOST.name, res3.trackList[0].status)

        // 4. DELETED after 6 missed frames
        for (i in 1..5) {
            multiHitBridge.processFrame(timestamp = 60.066 + i * 0.033, observations = emptyList())
        }
        val resPruned = multiHitBridge.processFrame(timestamp = 60.300, observations = emptyList())
        assertEquals(0, resPruned.activeTracks)
    }

    @Test
    fun testCompactVolumeMorphologyPassesValidation() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val cluster = CandidateCluster(
            minCol = 14,
            maxCol = 18,
            minRow = 10,
            maxRow = 20,
            sumCol = 16L * 45,
            sumRow = 15L * 45,
            size = 45,
            totalEdgeEnergy = 1800f,
            leftBottomRow = 20,
            centerBottomRow = 20,
            rightBottomRow = 20,
            fillRatio = 45.0 / (11 * 5),
            shapeClass = CandidateShapeClass.COMPACT_VOLUME,
            floorStructureLikelihood = 0.10,
            groundPlaneCoherence = 1.0
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = cluster,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertTrue("Compact volumetric obstacle should be valid", result.valid)
        assertEquals(CandidateShapeClass.COMPACT_VOLUME, result.shapeClass)
        assertTrue("Compact volume should receive confidence boost (>0.75)", result.score >= 0.75)
    }

    @Test
    fun testPlanarFloorSeamMorphologyRejected() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val cluster = CandidateCluster(
            minCol = 8,
            maxCol = 24,
            minRow = 20,
            maxRow = 21,
            sumCol = 16L * 18,
            sumRow = 20L * 18,
            size = 18,
            totalEdgeEnergy = 720f,
            leftBottomRow = 21,
            centerBottomRow = 21,
            rightBottomRow = 21,
            fillRatio = 18.0 / (2 * 17),
            shapeClass = CandidateShapeClass.PLANAR_FLOOR_FEATURE,
            floorStructureLikelihood = 0.85,
            groundPlaneCoherence = 1.0
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = cluster,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertFalse("Planar floor feature must be rejected", result.valid)
        assertTrue("Rejection reasons must include INVALID_HEIGHT or PLANAR_FLOOR_FEATURE", 
            result.reasons.contains(GeometryRejectReason.INVALID_HEIGHT) || 
            result.reasons.contains(GeometryRejectReason.PLANAR_FLOOR_FEATURE)
        )
    }

    @Test
    fun testLowVolumetricObstacleAccepted() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        // Low box on floor (H ~ 0.26m) with compact volumetric structure
        val cluster = CandidateCluster(
            minCol = 14,
            maxCol = 18,
            minRow = 19,
            maxRow = 21,
            sumCol = 16L * 15,
            sumRow = 20L * 15,
            size = 15,
            totalEdgeEnergy = 600f,
            leftBottomRow = 21,
            centerBottomRow = 21,
            rightBottomRow = 21,
            fillRatio = 15.0 / (3 * 5),
            shapeClass = CandidateShapeClass.COMPACT_VOLUME,
            floorStructureLikelihood = 0.10,
            groundPlaneCoherence = 1.0
        )
        val result = analyzer.validateCandidateGeometry(
            cluster = cluster,
            fx = 608.0,
            fy = 456.0,
            cx = 320.0,
            cy = 240.0,
            stepX = 20,
            stepY = 20
        )
        assertTrue("Low volumetric obstacle (H in 0.22..0.32m) should be accepted", result.valid)
        assertEquals(CandidateShapeClass.COMPACT_VOLUME, result.shapeClass)
        assertTrue("Physical height should be volumetric (~0.22..0.32m)", result.physicalHeightMeters in 0.20..0.35)
    }

    @Test
    fun testGroundPlaneCoherenceEvaluation() {
        val coherentCluster = CandidateCluster(
            minCol = 12, maxCol = 20, minRow = 14, maxRow = 20, sumCol = 16L * 25, sumRow = 17L * 25, size = 25,
            leftBottomRow = 20, centerBottomRow = 20, rightBottomRow = 20,
            fillRatio = 0.50, shapeClass = CandidateShapeClass.COMPACT_VOLUME, floorStructureLikelihood = 0.15, groundPlaneCoherence = 1.0
        )
        val slantedGlareCluster = CandidateCluster(
            minCol = 12, maxCol = 20, minRow = 14, maxRow = 20, sumCol = 16L * 25, sumRow = 17L * 25, size = 25,
            leftBottomRow = 15, centerBottomRow = 18, rightBottomRow = 21, // 6 row slant across floor
            fillRatio = 0.30, shapeClass = CandidateShapeClass.UNCERTAIN, floorStructureLikelihood = 0.55, groundPlaneCoherence = 0.20
        )

        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val resCoherent = analyzer.validateCandidateGeometry(coherentCluster, 608.0, 456.0, 320.0, 240.0, 20, 20)
        val resSlanted = analyzer.validateCandidateGeometry(slantedGlareCluster, 608.0, 456.0, 320.0, 240.0, 20, 20)

        assertTrue("Coherent candidate should be valid", resCoherent.valid)
        assertTrue("Coherent candidate should score higher than slanted glare", resCoherent.score > resSlanted.score)
    }

    @Test
    fun testSingleTrackAssociationAndVelocitySanity() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Approaching person at ~1.0 m/s
        val res1 = testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 3.0, className = "person")))
        assertEquals(1, res1.activeTracks)
        val trkId = res1.trackList[0].trackId

        val res2 = testBridge.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 2.9, className = "person")))
        assertEquals(trkId, res2.trackList[0].trackId)
        assertTrue("Vz should be negative indicating closing motion", res2.trackList[0].vz < 0.0)
        assertTrue("Vz should be within realistic physical bounds (-3.5..0.0)", res2.trackList[0].vz in -3.5..0.0)
    }

    @Test
    fun testTwoTrackAssociationAndCrossing() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Two persons on opposite sides
        val res1 = testBridge.processFrame(0.0, listOf(
            PrismObservation(x = -0.8, z = 2.5, className = "person"),
            PrismObservation(x = 0.8, z = 2.5, className = "person")
        ))
        assertEquals(2, res1.activeTracks)
        val idLeft = res1.trackList.first { it.x < 0 }.trackId
        val idRight = res1.trackList.first { it.x > 0 }.trackId

        // Frame 2: Moving toward centerline
        val res2 = testBridge.processFrame(0.1, listOf(
            PrismObservation(x = -0.5, z = 2.5, className = "person"),
            PrismObservation(x = 0.5, z = 2.5, className = "person")
        ))
        assertEquals(2, res2.activeTracks)
        assertEquals(idLeft, res2.trackList.first { it.vx > 0 }.trackId)
        assertEquals(idRight, res2.trackList.first { it.vx < 0 }.trackId)
    }

    @Test
    fun testTemporaryOcclusionAndReacquisition() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1-2: Track confirmed
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.2, z = 2.0, className = "person")))
        val res2 = testBridge.processFrame(0.033, listOf(PrismObservation(x = 0.2, z = 2.0, className = "person")))
        val trkId = res2.trackList[0].trackId
        assertEquals(TrackStatus.CONFIRMED.name, res2.trackList[0].status)

        // Frame 3: Temporary occlusion (empty observation) -> track is in LOST status
        val res3 = testBridge.processFrame(0.066, emptyList())
        assertEquals("Track should transition to LOST during occlusion", "LOST", res3.trackList[0].status)

        // Frame 4: Reacquired within gate -> retains identity and returns to CONFIRMED
        val res4 = testBridge.processFrame(0.099, listOf(PrismObservation(x = 0.2, z = 2.0, className = "person")))
        assertEquals(1, res4.activeTracks)
        assertEquals(trkId, res4.trackList[0].trackId)
        assertEquals(TrackStatus.CONFIRMED.name, res4.trackList[0].status)
    }

    @Test
    fun testVelocitySpikeClamping() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Stable origin
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 2.0, className = "person")))

        // Frame 2: Massive coordinate jump of 2.0m in 0.033s (apparent ~60 m/s)
        val res2 = testBridge.processFrame(0.033, listOf(PrismObservation(x = 2.0, z = 2.0, className = "person")))
        val vx = res2.trackList[0].vx
        assertTrue("Vx must be clamped to physical bounds (<= 3.5 m/s)", abs(vx) <= 3.5)
    }

    @Test
    fun testApproachingVsRecedingObstacleRisk() {
        val testBridgeApproaching = DeterministicPrismBridge()
        testBridgeApproaching.initialize()
        // Approaching person
        testBridgeApproaching.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 2.2, className = "person")))
        val resApproach = testBridgeApproaching.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 2.0, className = "person")))

        val testBridgeReceding = DeterministicPrismBridge()
        testBridgeReceding.initialize()
        // Receding person (moving away)
        testBridgeReceding.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 2.0, className = "person")))
        val resReceding = testBridgeReceding.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 2.2, className = "person")))

        assertTrue("Approaching obstacle must have higher risk than receding obstacle",
            resApproach.risk.riskScore > resReceding.risk.riskScore)
        assertEquals("Receding obstacle should yield safe/low risk or continue", "continue", resReceding.decision.action)
    }

    @Test
    fun testCrossingPedestrianTrajectoryCpa() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Pedestrian crossing laterally across corridor
        testBridge.processFrame(0.0, listOf(PrismObservation(x = -0.8, z = 2.0, className = "person")))
        val res = testBridge.processFrame(0.1, listOf(PrismObservation(x = -0.6, z = 2.0, className = "person")))

        val trk = res.trackList[0]
        assertTrue("Vx should be positive indicating rightward crossing motion", trk.vx > 0.0)
        assertTrue("Time to CPA should be finite and positive", trk.timeToCpa > 0.0)
    }

    @Test
    fun testDecisionDirectionHysteresis() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Obstacle to left (x = -0.30), evasion is turn_right
        val res1 = testBridge.processFrame(0.0, listOf(PrismObservation(x = -0.30, z = 2.0, className = "obstacle")))
        assertEquals("turn_right", res1.decision.action)

        // Frame 2: Slight noise shifts position to x = +0.05 near center
        // Directional hysteresis preserves turn_right
        val res2 = testBridge.processFrame(0.033, listOf(PrismObservation(x = 0.05, z = 2.0, className = "obstacle")))
        assertEquals("Hysteresis must preserve previous turn direction near centerline", "turn_right", res2.decision.action)
    }

    @Test
    fun testEmergencyStopPriorityOverHysteresis() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Step 1: Normal turn evasion
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.3, z = 2.0, className = "obstacle")))

        // Step 2: Immediate close obstacle (<1.0m)
        val resClose = testBridge.processFrame(0.033, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertEquals("Emergency stop must have absolute priority", "emergency_stop", resClose.decision.action)
        assertEquals("critical", resClose.decision.riskLevel)
    }

    @Test
    fun testWarningDeduplication() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Close hazard triggers speech cue
        val res1 = testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 0.8, className = "person")))
        assertTrue("First frame must emit speech alert", res1.cues.any { it.channel == "speech" })

        // Frame 2 (33ms later): Same hazard state does not duplicate speech alert within cooldown
        val res2 = testBridge.processFrame(0.033, listOf(PrismObservation(x = 0.0, z = 0.8, className = "person")))
        assertFalse("Frame within cooldown must not duplicate speech alert", res2.cues.any { it.channel == "speech" })

        // Frame 3 (1.6s later): After cooldown expires, speech alert can be repeated
        val res3 = testBridge.processFrame(1.6, listOf(PrismObservation(x = 0.0, z = 0.8, className = "person")))
        assertTrue("Frame after cooldown must emit speech alert", res3.cues.any { it.channel == "speech" })
    }

    @Test
    fun testGravityPitchAndRollEstimation() {
        val fusion = com.prism.assistant.sensor.OrientationFusion(
            calibration = com.prism.assistant.sensor.CameraMountCalibration(pitchOffsetRadians = 0.0, rollOffsetRadians = 0.0),
            mode = com.prism.assistant.sensor.OrientationMode.IMU_FUSED_ORIENTATION
        )

        // 1. Nominal upright orientation (gravity pointing along device Y axis: gx=0, gy=9.81, gz=0)
        fusion.processGravitySample(gx = 0.0, gy = 9.81, gz = 0.0, timestampNs = 1_000_000_000L)
        val oriUpright = fusion.getOrientation(1_000_000_000L)
        assertEquals(0.0, oriUpright.pitchRadians, 0.02)
        assertEquals(0.0, oriUpright.rollRadians, 0.02)
        assertEquals(com.prism.assistant.sensor.OrientationSourceState.GRAVITY_ONLY, oriUpright.sourceState)

        // 2. Forward pitch tilt by ~15 deg (gz > 0)
        val pitchRad15 = Math.toRadians(15.0)
        val gy15 = 9.81 * kotlin.math.cos(pitchRad15)
        val gz15 = 9.81 * kotlin.math.sin(pitchRad15)
        for (i in 1..10) {
            fusion.processGravitySample(gx = 0.0, gy = gy15, gz = gz15, timestampNs = 1_000_000_000L + i * 20_000_000L)
        }
        val oriPitched = fusion.getOrientation(1_200_000_000L)
        assertTrue("Pitch should reflect downward tilt (~0.26 rad)", oriPitched.pitchRadians in 0.15..0.35)

        // 3. Right roll tilt by ~10 deg (gx < 0)
        val rollRad10 = Math.toRadians(10.0)
        val gx10 = -9.81 * kotlin.math.sin(rollRad10)
        val gyRoll = 9.81 * kotlin.math.cos(rollRad10)
        for (i in 1..10) {
            fusion.processGravitySample(gx = gx10, gy = gyRoll, gz = 0.0, timestampNs = 1_300_000_000L + i * 20_000_000L)
        }
        val oriRolled = fusion.getOrientation(1_500_000_000L)
        assertTrue("Roll should be positive for rightward roll (~0.17 rad)", oriRolled.rollRadians > 0.05)
    }

    @Test
    fun testGyroIntegrationConstantRate() {
        val fusion = com.prism.assistant.sensor.OrientationFusion(
            calibration = com.prism.assistant.sensor.CameraMountCalibration(pitchOffsetRadians = 0.20, rollOffsetRadians = 0.0),
            mode = com.prism.assistant.sensor.OrientationMode.IMU_FUSED_ORIENTATION
        )

        // Initial sample
        fusion.processGyroSample(wx = 0.0, wy = 0.0, wz = 0.0, timestampNs = 1_000_000_000L)

        // Constant pitch rate of 0.5 rad/s for 100ms (dt = 0.1s) -> expected dPitch = 0.05 rad
        fusion.processGyroSample(wx = 0.50, wy = 0.0, wz = 0.0, timestampNs = 1_100_000_000L)
        val ori = fusion.getOrientation(1_100_000_000L)
        assertEquals("Pitch should increase by integrated angular velocity", 0.25, ori.pitchRadians, 0.01)
    }

    @Test
    fun testComplementaryFusionConvergence() {
        val fusion = com.prism.assistant.sensor.OrientationFusion(
            calibration = com.prism.assistant.sensor.CameraMountCalibration(pitchOffsetRadians = 0.20, rollOffsetRadians = 0.0),
            mode = com.prism.assistant.sensor.OrientationMode.IMU_FUSED_ORIENTATION
        )

        // 1. Initial gravity sample at pitch = 0.20 rad
        fusion.processGravitySample(0.0, 9.81, 0.0, 1_000_000_000L)

        // 2. High-speed gyro angular step: +0.15 rad
        fusion.processGyroSample(0.0, 0.0, 0.0, 1_000_000_000L)
        fusion.processGyroSample(1.5, 0.0, 0.0, 1_100_000_000L) // +0.15 rad
        val oriFast = fusion.getOrientation(1_100_000_000L)
        assertTrue("Fusion must follow rapid gyro angular motion", oriFast.pitchRadians > 0.30)
        assertEquals(com.prism.assistant.sensor.OrientationSourceState.IMU_FUSED, oriFast.sourceState)

        // 3. Steady gravity samples correct slow drift
        for (i in 1..25) {
            fusion.processGravitySample(0.0, 9.81, 0.0, 1_100_000_000L + i * 40_000_000L)
        }
        val oriCorrected = fusion.getOrientation(2_100_000_000L)
        assertTrue("Fusion must slowly correct toward gravity reference", oriCorrected.pitchRadians < oriFast.pitchRadians)
    }

    @Test
    fun testAbnormalGravityAndNanProtection() {
        val fusion = com.prism.assistant.sensor.OrientationFusion()
        val initialOri = fusion.getOrientation()

        // Feed NaN and infinite vectors
        fusion.processGravitySample(Double.NaN, 9.81, 0.0, 1_000_000_000L)
        fusion.processGyroSample(0.0, Double.POSITIVE_INFINITY, 0.0, 1_000_000_000L)
        // Feed abnormal gravity magnitude (<1.0 or >25.0)
        fusion.processGravitySample(0.0, 0.05, 0.0, 1_100_000_000L)

        val safeOri = fusion.getOrientation()
        assertFalse("Pitch must not be NaN", safeOri.pitchRadians.isNaN())
        assertFalse("Roll must not be NaN", safeOri.rollRadians.isNaN())
        assertEquals(initialOri.pitchRadians, safeOri.pitchRadians, 0.01)
    }

    @Test
    fun testOrientationModesAndMountCalibration() {
        val cal = com.prism.assistant.sensor.CameraMountCalibration(pitchOffsetRadians = 0.20944, rollOffsetRadians = 0.05)
        val fusion = com.prism.assistant.sensor.OrientationFusion(calibration = cal)

        // 1. Fixed Mode always outputs calibrated constants
        fusion.mode = com.prism.assistant.sensor.OrientationMode.FIXED_ORIENTATION
        fusion.processGyroSample(1.0, 1.0, 0.0, 1_100_000_000L)
        val fixedOri = fusion.getOrientation()
        assertEquals(0.20944, fixedOri.pitchRadians, 0.001)
        assertEquals(0.05, fixedOri.rollRadians, 0.001)
        assertEquals(com.prism.assistant.sensor.OrientationSourceState.FALLBACK_CALIBRATED, fixedOri.sourceState)

        // 2. Fused Mode incorporates dynamic tilt
        fusion.mode = com.prism.assistant.sensor.OrientationMode.IMU_FUSED_ORIENTATION
        val fusedOri = fusion.getOrientation()
        assertEquals(com.prism.assistant.sensor.OrientationSourceState.FALLBACK_CALIBRATED, fusedOri.sourceState)
    }

    @Test
    fun testZeroRollRegressionMatchesFixedProjection() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val cluster = CandidateCluster(
            minCol = 14, maxCol = 18, minRow = 10, maxRow = 20,
            sumCol = 16L * 15, sumRow = 15L * 15, size = 15, totalEdgeEnergy = 600f
        )

        // 1. Default validation (zero roll)
        val resDefault = analyzer.validateCandidateGeometry(
            cluster = cluster, fx = 608.0, fy = 456.0, cx = 320.0, cy = 240.0, stepX = 20, stepY = 20
        )

        // 2. Explicit zero-roll validation
        val resExplicitZeroRoll = analyzer.validateCandidateGeometry(
            cluster = cluster, fx = 608.0, fy = 456.0, cx = 320.0, cy = 240.0, stepX = 20, stepY = 20,
            pitchRadians = 0.20944, rollRadians = 0.0
        )

        assertTrue(resDefault.valid)
        assertTrue(resExplicitZeroRoll.valid)
        assertEquals("Ground Z must match perfectly in zero roll", resDefault.groundZ, resExplicitZeroRoll.groundZ, 1e-6)
        assertEquals("Ground X must match perfectly in zero roll", resDefault.groundX, resExplicitZeroRoll.groundX, 1e-6)
        assertEquals("Physical Height must match perfectly in zero roll", resDefault.physicalHeightMeters, resExplicitZeroRoll.physicalHeightMeters, 1e-6)
    }

    @Test
    fun testNonZeroRollProjectionCompensation() {
        val analyzer = AndroidPerceptionAnalyzer(PerceptionConfig())
        val cluster = CandidateCluster(
            minCol = 14, maxCol = 18, minRow = 12, maxRow = 20,
            sumCol = 16L * 15, sumRow = 16L * 15, size = 15, totalEdgeEnergy = 600f
        )

        // 1. Level zero roll
        val resLevel = analyzer.validateCandidateGeometry(
            cluster = cluster, fx = 608.0, fy = 456.0, cx = 320.0, cy = 240.0, stepX = 20, stepY = 20,
            pitchRadians = 0.20944, rollRadians = 0.0
        )

        // 2. Right roll tilt (+10 degrees = +0.1745 rad)
        val resRollRight = analyzer.validateCandidateGeometry(
            cluster = cluster, fx = 608.0, fy = 456.0, cx = 320.0, cy = 240.0, stepX = 20, stepY = 20,
            pitchRadians = 0.20944, rollRadians = Math.toRadians(10.0)
        )

        // 3. Left roll tilt (-10 degrees = -0.1745 rad)
        val resRollLeft = analyzer.validateCandidateGeometry(
            cluster = cluster, fx = 608.0, fy = 456.0, cx = 320.0, cy = 240.0, stepX = 20, stepY = 20,
            pitchRadians = 0.20944, rollRadians = Math.toRadians(-10.0)
        )

        assertTrue("Level projection should be valid", resLevel.valid)
        assertTrue("Roll-compensated projection (right) should be valid", resRollRight.valid)
        assertTrue("Roll-compensated projection (left) should be valid", resRollLeft.valid)
        assertTrue("Roll right shifts lateral coordinate left relative to optical axis", resRollRight.groundX < resLevel.groundX)
        assertTrue("Roll left shifts lateral coordinate right relative to optical axis", resRollLeft.groundX > resLevel.groundX)
    }

    @Test
    fun testLargeTimestampGapGuard() {
        val fusion = com.prism.assistant.sensor.OrientationFusion()
        fusion.processGyroSample(0.0, 0.0, 0.0, 1_000_000_000L)

        // Giant 10 second gap with non-zero rate -> should be rejected without exploding
        fusion.processGyroSample(1.0, 0.0, 0.0, 11_000_000_000L)
        val ori = fusion.getOrientation(11_000_000_000L)
        assertTrue("Pitch must remain bounded within physical human range", ori.pitchRadians in -0.35..0.85)
    }

    // =========================================================================
    // Phase H — End-to-End System Release & Accessibility Validation Tests
    // =========================================================================

    @Test
    fun testAudioPriorityArbitrationAndPreemption() {
        val controller = AccessibilityController()
        
        // 1. Dispatch Emergency stop
        val emergencyDispatched = controller.dispatchSpeech("Stop", AudioPriority.EMERGENCY)
        assertTrue("Emergency cue should be dispatched immediately", emergencyDispatched)

        // 2. Immediate low-priority navigation cue should be suppressed (<1.0s)
        val routineDispatched = controller.dispatchSpeech("Proceed forward along corridor", AudioPriority.NAVIGATION)
        assertFalse("Routine navigation should not interrupt recent emergency alert", routineDispatched)

        // 3. Higher priority rank ordering validation
        assertTrue(AudioPriority.EMERGENCY.rank > AudioPriority.SAFETY_WARNING.rank)
        assertTrue(AudioPriority.SAFETY_WARNING.rank > AudioPriority.NAVIGATION.rank)
        assertTrue(AudioPriority.NAVIGATION.rank > AudioPriority.ADVISORY.rank)
        assertTrue(AudioPriority.ADVISORY.rank > AudioPriority.SYSTEM_INFO.rank)
    }

    @Test
    fun testAccessibleQueriesLandmarkAndObstacles() {
        val controller = AccessibilityController()

        // 1. Where am I query
        val whereText = controller.queryWhereAmI()
        assertTrue("Should contain corridor status", whereText.contains("corridor") || whereText.contains("concourse"))
        assertTrue("Should reference default destination", whereText.contains("North Concourse Exit"))

        // 2. What is ahead query - clear corridor
        val clearAhead = controller.queryWhatIsAhead(null)
        assertEquals("Path ahead is clear.", clearAhead)

        // 3. What is ahead query - active emergency stop
        val bridge = DeterministicPrismBridge()
        bridge.initialize()
        val obs = listOf(PrismObservation(x = 0.0, z = 0.8, className = "person"))
        val hazardResult = bridge.processFrame(timestamp = 1.0, observations = obs)
        val hazardAhead = controller.queryWhatIsAhead(hazardResult)
        assertEquals("Obstacle directly ahead. Stop.", hazardAhead)

        // 4. Repeat last instruction
        val repeated = controller.repeatLastInstruction()
        assertNotNull(repeated)
        assertTrue(repeated.isNotEmpty())
    }

    @Test
    fun testDestinationSelectionFlow() {
        val controller = AccessibilityController()
        assertEquals("D1", controller.selectedDestination?.id)
        
        // Select elevator platform D3
        val msg = controller.selectDestination("D3")
        assertEquals("D3", controller.selectedDestination?.id)
        assertEquals("Elevator to All Platforms", controller.selectedDestination?.name)
        assertEquals(NavigationState.STARTING, controller.navigationState)
        assertTrue(msg.contains("Elevator to All Platforms"))
    }

    @Test
    fun testSafeDegradationHumanReadableSpeech() {
        val controller = AccessibilityController()
        assertEquals(SafeDegradationState.FULL, controller.safeDegradationState)

        // Sensor unavailable degradation
        val state1 = controller.updateSystemHealthState(
            sourceState = OrientationSourceState.UNAVAILABLE,
            cameraHealthy = true,
            pipelineRunning = true
        )
        assertEquals(SafeDegradationState.DEGRADED_SENSOR, state1)
        assertEquals(SafeDegradationState.DEGRADED_SENSOR, controller.safeDegradationState)

        // Gyro fallback calibrated degradation
        val state2 = controller.updateSystemHealthState(
            sourceState = OrientationSourceState.FALLBACK_CALIBRATED,
            cameraHealthy = true,
            pipelineRunning = true
        )
        assertEquals(SafeDegradationState.DEGRADED_ORIENTATION, state2)

        // Camera hardware failure
        val state3 = controller.updateSystemHealthState(
            sourceState = OrientationSourceState.IMU_FUSED,
            cameraHealthy = false,
            pipelineRunning = true
        )
        assertEquals(SafeDegradationState.UNAVAILABLE, state3)
    }

    @Test
    fun testReleaseModeSeparation() {
        val controller = AccessibilityController()
        // Default mode must be PARTICIPANT_STUDY for BVI accessibility
        assertEquals(ReleaseMode.PARTICIPANT_STUDY, controller.releaseMode)

        controller.releaseMode = ReleaseMode.RESEARCH_DEBUG
        assertEquals(ReleaseMode.RESEARCH_DEBUG, controller.releaseMode)
    }

    @Test
    fun testHapticLanguagePatterns() {
        val controller = AccessibilityController()
        assertTrue(controller.dispatchHaptic("emergency", "emergency"))
        assertTrue(controller.dispatchHaptic("slow", "warning"))
        assertTrue(controller.dispatchHaptic("turn_left", "warning"))
        assertTrue(controller.dispatchHaptic("turn_right", "warning"))
    }

    @Test
    fun testBaselineConfigFreezePRISMH1() {
        assertEquals("PRISM-H1.0.0-RELEASE", PrismBaselineConfig.RELEASE_VERSION)
        assertEquals(1.30, PrismBaselineConfig.CAMERA_HEIGHT_METERS, 1e-4)
        assertEquals(0.20944, PrismBaselineConfig.NOMINAL_PITCH_RADIANS, 1e-4)
        assertEquals(0.32, PrismBaselineConfig.MIN_PHYSICAL_HEIGHT_M, 1e-4)
        assertEquals(0.22, PrismBaselineConfig.MIN_VOLUMETRIC_HEIGHT_M, 1e-4)
        assertEquals(2.40, PrismBaselineConfig.MAX_PHYSICAL_HEIGHT_M, 1e-4)
        assertEquals(0.40, PrismBaselineConfig.MIN_DEPTH_M, 1e-4)
        assertEquals(6.50, PrismBaselineConfig.MAX_DEPTH_M, 1e-4)
        assertEquals(1.25, PrismBaselineConfig.MAX_LATERAL_EXTENT_M, 1e-4)
        assertEquals(2, PrismBaselineConfig.CONFIRMATION_HITS)
        assertEquals(1.00, PrismBaselineConfig.EMERGENCY_STOP_THRESHOLD_M, 1e-4)
        assertTrue(PrismBaselineConfig.AUDIO_DUCKING_ENABLED)
    }

    @Test
    fun testCloseHazardZeroLatencyRegression() {
        val bridge = DeterministicPrismBridge(confirmationHits = 2)
        bridge.initialize()
        // Close hazard directly in front at 0.75m (within 1.0m emergency threshold)
        val obs = listOf(PrismObservation(x = 0.0, z = 0.75, className = "person"))
        val result = bridge.processFrame(timestamp = 0.0, observations = obs)
        
        // Immediate emergency stop on frame 0 without waiting for second hit
        assertEquals("emergency_stop", result.decision.action)
        assertEquals("critical", result.decision.riskLevel)
        assertTrue(result.cues.any { it.channel == "speech" && (it.text == "Stop" || it.text == "Stop. Obstacle directly ahead.") })
        assertTrue(result.cues.any { it.channel == "haptic" && it.hapticPattern == "emergency" })
    }

    // =========================================================================
    // Warning System Redesign — Spatial, Graduated and Context-Aware Tests
    // =========================================================================

    @Test
    fun testWarningSystemClearSceneNoRepetitiveSpeech() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 5 consecutive clear frames -> 0 speech alerts emitted
        var totalSpeechCues = 0
        for (i in 0 until 5) {
            val res = testBridge.processFrame(timestamp = i * 0.033, observations = emptyList())
            assertEquals("continue", res.decision.action)
            assertEquals("safe", res.decision.riskLevel)
            totalSpeechCues += res.cues.count { it.channel == "speech" }
        }
        assertEquals("Clear scene must not emit repetitive speech", 0, totalSpeechCues)
    }

    @Test
    fun testWarningSystemAwarenessLowRisk() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Stationary person in corridor at 4.2m
        val obs = listOf(PrismObservation(x = 0.50, z = 4.2, className = "person"))
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        assertEquals("continue", res.decision.action)
        val speechCue = res.cues.find { it.channel == "speech" }
        assertNotNull("Awareness cue should be emitted", speechCue)
        assertTrue("Speech should indicate person ahead", speechCue!!.text!!.contains("Person ahead"))
        // Aware state should have no haptic vibration
        assertFalse("Awareness level should not trigger haptics", res.cues.any { it.channel == "haptic" })
    }

    @Test
    fun testWarningSystemDirectionalMoveRight() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Obstacle on left (x = -0.35m, z = 2.2m) with clear right corridor
        val obs = listOf(PrismObservation(x = -0.35, z = 2.2, className = "obstacle"))
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        assertEquals("turn_right", res.decision.action)
        val speechCue = res.cues.find { it.channel == "speech" }
        assertNotNull("Directional guidance cue should be emitted", speechCue)
        assertTrue("Should recommend moving right", speechCue!!.text!!.contains("Move right"))
        assertTrue("Directional right should trigger right haptic", res.cues.any { it.channel == "haptic" && it.hapticPattern == "right" })
    }

    @Test
    fun testWarningSystemCenteredThreatDirectlyAhead() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Centered obstacle at 3.0m
        val obs = listOf(PrismObservation(x = 0.0, z = 3.0, className = "obstacle"))
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        val speechCue = res.cues.find { it.channel == "speech" }
        assertNotNull("Speech cue should be emitted", speechCue)
        assertTrue("Centered obstacle should announce directly ahead", speechCue!!.text!!.contains("directly ahead") || speechCue.text!!.contains("three meters"))
    }

    @Test
    fun testWarningSystemEmergencyCriticalThreat() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Critical close hazard at 0.7m
        val obs = listOf(PrismObservation(x = 0.0, z = 0.7, className = "person"))
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        assertEquals("emergency_stop", res.decision.action)
        assertEquals("critical", res.decision.riskLevel)
        val speechCue = res.cues.find { it.channel == "speech" }
        assertNotNull(speechCue)
        assertTrue("Emergency cue must start with Stop", speechCue!!.text!!.startsWith("Stop"))
        assertTrue("Emergency haptic required", res.cues.any { it.channel == "haptic" && it.hapticPattern == "emergency" })
    }

    @Test
    fun testWarningSystemRecedingObjectNoEmergency() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Target moving away (positive vz): frame 1 at z=2.5, frame 2 at z=3.0 (vz ~ +0.5 m/s)
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")))
        val res2 = testBridge.processFrame(timestamp = 1.0, observations = listOf(PrismObservation(x = 0.0, z = 3.0, className = "person")))

        // Moving away should NOT produce emergency stop
        assertFalse("Receding object must not trigger emergency stop", res2.decision.action == "emergency_stop")
        val speechCue = res2.cues.find { it.channel == "speech" }
        if (speechCue != null) {
            assertFalse("Receding object speech must not say Stop", speechCue.text!!.contains("Stop"))
        }
    }

    @Test
    fun testWarningSystemCrossingPersonDirectional() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Person crossing from left to right:
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = -0.7, z = 2.4, className = "person")))
        val res2 = testBridge.processFrame(timestamp = 1.6, observations = listOf(PrismObservation(x = -0.3, z = 2.4, className = "person")))

        val speechCue = res2.cues.find { it.channel == "speech" }
        assertNotNull("Crossing cue should be emitted", speechCue)
        assertTrue("Should announce crossing from left", speechCue!!.text!!.contains("crossing from left"))
    }

    @Test
    fun testWarningSystemAmbiguousDirectionBothSidesBlocked() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Direct obstacle ahead with both left and right corridor obstructed
        val obs = listOf(
            PrismObservation(x = 0.0, z = 1.4, className = "obstacle"),
            PrismObservation(x = -0.4, z = 1.5, className = "obstacle"),
            PrismObservation(x = 0.4, z = 1.5, className = "obstacle")
        )
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        val speechCue = res.cues.find { it.channel == "speech" }
        assertNotNull("Speech alert should be emitted", speechCue)
        assertTrue("When both sides blocked, path blocked or stop required", 
            speechCue!!.text!!.contains("Path blocked") || speechCue.text!!.startsWith("Stop"))
    }

    @Test
    fun testWarningSystemMultiObjectSinglePriorityWarning() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Two objects: one distant at 4.5m, one urgent at 1.4m
        val obs = listOf(
            PrismObservation(x = 0.8, z = 4.5, className = "obstacle"),
            PrismObservation(x = -0.3, z = 1.4, className = "person")
        )
        val res = testBridge.processFrame(timestamp = 0.0, observations = obs)

        val speechCues = res.cues.filter { it.channel == "speech" }
        assertTrue("At most 1 speech alert per frame to avoid chatter", speechCues.size <= 1)
        if (speechCues.isNotEmpty()) {
            assertTrue(speechCues[0].text!!.contains("Person") || speechCues[0].text!!.contains("close") || speechCues[0].text!!.contains("Move"))
        }
    }

    @Test
    fun testWarningSystemEscalationAndDeescalation() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 1. Far away -> AWARE
        val res1 = testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 4.5, className = "obstacle")))
        val speech1 = res1.cues.find { it.channel == "speech" }?.text
        assertNotNull(speech1)

        // 2. Escalation to Caution (3.0m)
        val res2 = testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 3.0, className = "obstacle")))
        val speech2 = res2.cues.find { it.channel == "speech" }?.text
        assertNotNull("Escalation to caution should emit immediately", speech2)

        // 3. Escalation to Emergency (0.7m)
        val res3 = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 0.7, className = "obstacle")))
        val speech3 = res3.cues.find { it.channel == "speech" }?.text
        assertNotNull(speech3)
        assertTrue(speech3!!.startsWith("Stop"))

        // 4. De-escalation to Clear: Feed empty frames until track coasting expires and path clears
        var speech4: String? = null
        for (f in 1..7) {
            val res = testBridge.processFrame(timestamp = 0.2 + f * 0.05, observations = emptyList())
            val text = res.cues.find { it.channel == "speech" }?.text
            if (text != null) {
                speech4 = text
            }
        }
        assertEquals("Path clear.", speech4)

        // 5. Subsequent clear frame -> SILENT
        val res5 = testBridge.processFrame(timestamp = 1.0, observations = emptyList())
        val speech5 = res5.cues.find { it.channel == "speech" }?.text
        assertEquals("Subsequent clear frames must be silent", null, speech5)
    }

    @Test
    fun testSpatialMappingConfig() {
        val config = SpatialMappingConfig()
        assertEquals(SpatialDirection.FAR_LEFT, config.classify(-1.0))
        assertEquals(SpatialDirection.LEFT, config.classify(-0.5))
        assertEquals(SpatialDirection.SLIGHTLY_LEFT, config.classify(-0.25))
        assertEquals(SpatialDirection.CENTER, config.classify(0.0))
        assertEquals(SpatialDirection.SLIGHTLY_RIGHT, config.classify(0.25))
        assertEquals(SpatialDirection.RIGHT, config.classify(0.5))
        assertEquals(SpatialDirection.FAR_RIGHT, config.classify(1.0))
    }

    @Test
    fun testDistanceQuantization() {
        assertEquals(DistanceBand.CRITICAL, WarningMessageGenerator.quantizeDistance(0.5))
        assertEquals(DistanceBand.CLOSE, WarningMessageGenerator.quantizeDistance(1.2))
        assertEquals(DistanceBand.NEAR, WarningMessageGenerator.quantizeDistance(2.2))
        assertEquals(DistanceBand.MEDIUM, WarningMessageGenerator.quantizeDistance(3.2))
        assertEquals(DistanceBand.FAR, WarningMessageGenerator.quantizeDistance(4.2))
        assertEquals(DistanceBand.VERY_FAR, WarningMessageGenerator.quantizeDistance(5.2))
    }

    @Test
    fun testEmergencyHazardPersistsAfterUserStops() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Emergency stop alert for close obstacle at 0.8m
        val res1 = testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertEquals("emergency_stop", res1.decision.action)
        assertTrue(res1.cues.any { it.channel == "speech" && it.text!!.startsWith("Stop") })

        // Frame 2: User stopped, relative Vz ~ 0. Obstacle remains at 0.8m
        val res2 = testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertEquals("emergency_stop", res2.decision.action)
        assertEquals(1, res2.activeTracks)
        val trk = res2.trackList[0]
        assertEquals("HAZARD_PERSISTING", trk.hazardState)
        assertEquals("STOPPED", trk.userMotionState)
        // Must NOT downgrade to CAUTION, AWARE, or SLOW_DOWN
        assertFalse("Persistent hazard must not downgrade to caution", res2.decision.action == "slow_down" || res2.decision.action == "continue")
    }

    @Test
    fun testNoDuplicateEmergencyForPersistentHazard() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Frame 1: Emergency alert
        val res1 = testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertTrue(res1.cues.any { it.channel == "speech" && it.text!!.startsWith("Stop") })

        // Frames 2-4: User remains stopped in front of same obstacle
        val res2 = testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        val res3 = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        val res4 = testBridge.processFrame(timestamp = 0.5, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        assertTrue("Frame 2 speech must be suppressed in hazard hold", res2.cues.none { it.channel == "speech" })
        assertTrue("Frame 3 speech must be suppressed in hazard hold", res3.cues.none { it.channel == "speech" })
        assertTrue("Frame 4 speech must be suppressed in hazard hold", res4.cues.none { it.channel == "speech" })
    }

    @Test
    fun testHazardClearsAfterRiskReduction() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 1. Initial emergency
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // 2. User stopped, hazard persisting
        testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // 3. User steps backward or distance increases materially to 1.6m
        val res3 = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 1.6, className = "obstacle")))
        assertEquals("HAZARD_CLEARED", res3.trackList[0].hazardState)
        val clearSpeech = res3.cues.find { it.channel == "speech" }?.text
        assertEquals("Path clear.", clearSpeech)
    }

    @Test
    fun testSafetyStopDistinctFromUserPause() {
        val acc = AccessibilityController()

        // 1. User-initiated pause
        acc.navigationState = NavigationState.PAUSED
        val pauseMsg = acc.queryWhereAmI()
        assertEquals("Navigation paused in main corridor.", pauseMsg)

        // Safety stop must not override user-initiated PAUSED
        acc.setSafetyStop(true)
        assertEquals(NavigationState.PAUSED, acc.navigationState)

        // 2. Safety stop from navigating
        acc.navigationState = NavigationState.NAVIGATING
        acc.setSafetyStop(true)
        assertEquals(NavigationState.STOPPED_FOR_HAZARD, acc.navigationState)
        val safetyMsg = acc.queryWhereAmI()
        assertEquals("Stopped for obstacle. Path currently obstructed.", safetyMsg)

        // 3. Resuming after hazard clear
        acc.setSafetyStop(false)
        assertEquals(NavigationState.NAVIGATING, acc.navigationState)
    }

    @Test
    fun testNewHazardCanOverrideExistingHazardHold() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 1. Hazard A at 0.8m -> Emergency stop
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // 2. Hazard A held silent
        val res2 = testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertTrue(res2.cues.none { it.channel == "speech" })

        // 3. Hazard B appears at 0.7m while Hazard A is still held
        val res3 = testBridge.processFrame(
            timestamp = 0.2,
            observations = listOf(
                PrismObservation(x = 0.0, z = 0.8, className = "obstacle"),
                PrismObservation(x = 0.3, z = 0.7, className = "obstacle")
            )
        )
        // Must emit emergency alert for the new hazard
        assertTrue(
            "New hazard must be announced even while old hazard is held",
            res3.cues.any { it.channel == "speech" && it.text!!.contains("Stop") }
        )
    }

    @Test
    fun testHazardReEscalatesWhenUserResumesTowardObstacle() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 1. Initial emergency
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // 2. User stopped, hazard hold active
        val res2 = testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertEquals("HAZARD_PERSISTING", res2.trackList[0].hazardState)
        assertTrue(res2.cues.none { it.channel == "speech" })

        // 3. User resumes walking forward toward obstacle (distance drops to 0.65m)
        val res3 = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 0.65, className = "obstacle")))
        assertEquals("EMERGENCY", res3.trackList[0].hazardState)
        assertTrue(
            "Re-escalation must emit emergency stop warning again",
            res3.cues.any { it.channel == "speech" && it.text!!.startsWith("Stop") }
        )
    }

    @Test
    fun testMovingAwayHazardResolves() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // 1. Initial emergency at 0.9m
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.9, className = "person")))

        // 2. User stops
        testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.9, className = "person")))

        // 3. Person moves away forward down corridor (z increases with positive velocity)
        val res3 = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 1.35, className = "person")))
        assertEquals("HAZARD_CLEARED", res3.trackList[0].hazardState)
        assertTrue(res3.cues.any { it.channel == "speech" && it.text == "Path clear." })
    }

    @Test
    fun testPersistentStationaryHazardDoesNotBecomeClear() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // Run 10 frames where user and obstacle remain stationary
        for (i in 1..10) {
            val res = testBridge.processFrame(timestamp = i * 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
            assertEquals("emergency_stop", res.decision.action)
            assertEquals("HAZARD_PERSISTING", res.trackList[0].hazardState)
            assertFalse(
                "Persistent stationary obstacle must never announce Path clear",
                res.cues.any { it.channel == "speech" && it.text == "Path clear." }
            )
        }
    }

    @Test
    fun testPathClearAnnouncedOnceAfterResolution() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()

        // Emergency -> Hold -> Clear
        testBridge.processFrame(timestamp = 0.0, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        testBridge.processFrame(timestamp = 0.1, observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        val resClear = testBridge.processFrame(timestamp = 0.2, observations = listOf(PrismObservation(x = 0.0, z = 1.6, className = "obstacle")))
        assertTrue(resClear.cues.any { it.channel == "speech" && it.text == "Path clear." })

        // Subsequent frame must NOT announce Path clear again
        val resNext = testBridge.processFrame(timestamp = 0.3, observations = listOf(PrismObservation(x = 0.0, z = 1.6, className = "obstacle")))
        assertFalse(
            "Path clear must not be repeated in subsequent frames",
            resNext.cues.any { it.channel == "speech" && it.text == "Path clear." }
        )

        // Subsequent empty frame must also remain silent
        val resEmpty = testBridge.processFrame(timestamp = 0.4, observations = emptyList())
        assertFalse(
            "Clear corridor must remain silent",
            resEmpty.cues.any { it.channel == "speech" && it.text == "Path clear." }
        )
    }

    // =========================================================================
    // Phase J — Step Detection & IMU Ego-Motion Fusion Unit Tests
    // =========================================================================

    @Test
    fun testStepDetectionPeriodicWalking() {
        val estimator = UserMotionEstimator()
        // Simulate 8 steps at 1.8 Hz (period ~ 0.55s) with 50 Hz sampling (dt = 0.02s)
        val dtNs = 20_000_000L
        var currentTsNs = 1_000_000_000L
        val stepIntervalSec = 0.55
        val samplesPerStep = (stepIntervalSec / 0.02).toInt() // ~27 samples
        val numSteps = 8

        for (step in 0 until numSteps) {
            for (i in 0 until samplesPerStep) {
                currentTsNs += dtNs
                // Walking vertical acceleration profile: baseline gravity 9.8 + sinusoidal excursion
                val dynAcc = if (i in 4..10) 2.2 * sin((i - 4).toDouble() / 6.0 * Math.PI) else 0.1
                val ay = 9.80665 + dynAcc
                estimator.processAccelerometerSample(0.1, ay, 0.1, currentTsNs, 0.0, 9.80665, 0.0)
                estimator.processGyroSample(0.05, 0.05, 0.05, currentTsNs)
            }
        }

        val estimate = estimator.getMotionEstimate()
        assertTrue("Detected step count should be >= 6, was ${estimate.stepCount}", estimate.stepCount in 6..10)
        assertEquals(UserMotionState.WALKING, estimate.state)
        assertTrue("Walking confidence should be >= 0.80, was ${estimate.confidence}", estimate.confidence >= 0.80)
        assertTrue("Cadence should be positive", estimate.cadenceStepsPerMinute > 0.0)
    }

    private fun injectStepPulse(estimator: UserMotionEstimator, startTs: Long, peakDyn: Double = 2.2, samples: Int = 7): Long {
        var ts = startTs
        val dtNs = 20_000_000L
        for (i in 0 until samples) {
            ts += dtNs
            val dyn = peakDyn * sin((i.toDouble() / (samples - 1)) * Math.PI)
            estimator.processAccelerometerSample(0.0, 9.80665 + dyn, 0.0, ts, 0.0, 9.80665, 0.0)
        }
        for (i in 0 until 5) {
            ts += dtNs
            estimator.processAccelerometerSample(0.0, 9.80665, 0.0, ts, 0.0, 9.80665, 0.0)
        }
        return ts
    }

    @Test
    fun testStepRefractoryPeriod() {
        val estimator = UserMotionEstimator(StepDetectorConfig(refractoryPeriodSeconds = 0.28, stepThresholdMps2 = 1.20))
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Peak 1: valid step candidate
        tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)
        val countAfterFirst = estimator.getMotionEstimate().stepCount
        assertEquals(1, countAfterFirst)

        // Peak 2: rapid pulse only 60ms later (inside 280ms refractory period)
        for (i in 0 until 5) {
            tsNs += dtNs
            val dyn = 2.0 * sin(i.toDouble() / 4.0 * Math.PI)
            estimator.processAccelerometerSample(0.0, 9.80665 + dyn, 0.0, tsNs, 0.0, 9.80665, 0.0)
        }
        val countAfterRapid = estimator.getMotionEstimate().stepCount
        assertEquals("Refractory period must prevent rapid double-counting of a single physical event", 1, countAfterRapid)
    }

    @Test
    fun testArmSwingDoesNotCreateExcessiveSteps() {
        val estimator = UserMotionEstimator(StepDetectorConfig(minStepsForWalking = 2, stoppingTimeoutSeconds = 1.0))
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Single isolated arm swing excursion (1 peak)
        tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.0)
        val estMid = estimator.getMotionEstimate()
        assertEquals(1, estMid.stepCount)
        assertEquals(UserMotionState.STARTING, estMid.state) // Never WALKING because minStepsForWalking = 2

        // User remains still for 1.5 seconds following arm swing
        for (i in 0 until 75) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81 + 0.02, 0.0, tsNs, 0.0, 9.81, 0.0)
        }

        val estFinal = estimator.getMotionEstimate()
        assertEquals("Single arm swing must not produce multiple steps", 1, estFinal.stepCount)
        assertEquals("Single arm swing must decay back to stationary", UserMotionState.STATIONARY, estFinal.state)
    }

    @Test
    fun testStationaryAccelerationDoesNotCreateSteps() {
        val estimator = UserMotionEstimator()
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // 200 samples of stationary sensor with minor sensor noise
        for (i in 0 until 200) {
            tsNs += dtNs
            val noise = if (i % 2 == 0) 0.05 else -0.05
            estimator.processAccelerometerSample(0.02, 9.81 + noise, 0.01, tsNs, 0.0, 9.81, 0.0)
            estimator.processGyroSample(0.01, 0.01, 0.01, tsNs)
        }

        val est = estimator.getMotionEstimate()
        assertEquals(0, est.stepCount)
        assertEquals(UserMotionState.STATIONARY, est.state)
        assertTrue("Stationary confidence must be >= 0.85, was ${est.confidence}", est.confidence >= 0.85)
    }

    @Test
    fun testWalkingConfidenceBuilds() {
        val estimator = UserMotionEstimator()
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Step 1
        tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)
        for (i in 0 until 15) { tsNs += dtNs; estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0) }
        val conf1 = estimator.getMotionEstimate().confidence
        assertEquals(UserMotionState.STARTING, estimator.getMotionEstimate().state)

        // Step 2
        tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)
        for (i in 0 until 15) { tsNs += dtNs; estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0) }
        val conf2 = estimator.getMotionEstimate().confidence
        assertEquals(UserMotionState.WALKING, estimator.getMotionEstimate().state)
        assertTrue("Confidence must increase with second step", conf2 >= conf1)

        // Steps 3, 4, 5
        for (step in 0 until 3) {
            tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)
            for (i in 0 until 15) { tsNs += dtNs; estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0) }
        }
        val confFinal = estimator.getMotionEstimate().confidence
        assertTrue("Walking confidence should build to high certainty >= 0.90, was $confFinal", confFinal >= 0.90)
    }

    @Test
    fun testWalkingConfidenceDecays() {
        val estimator = UserMotionEstimator()
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Walking baseline (4 steps)
        for (step in 0 until 4) {
            tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)
            for (i in 0 until 10) { tsNs += dtNs; estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0) }
        }
        val initialWalkingEst = estimator.getMotionEstimate()
        assertEquals(UserMotionState.WALKING, initialWalkingEst.state)

        // Steps cease: 30 stationary samples (~0.6s) -> WALKING with decaying confidence
        for (i in 0 until 30) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        val decayEst = estimator.getMotionEstimate()
        assertTrue("Confidence should decay during pause between steps", decayEst.confidence <= initialWalkingEst.confidence)

        // Past 1.0s timeout (another 15 samples = 0.3s, total ~1.26s since last step) -> STOPPING
        for (i in 0 until 15) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        assertEquals(UserMotionState.STOPPING, estimator.getMotionEstimate().state)

        // Past 1.5s timeout (another 25 samples = 0.5s, total ~1.76s since last step) -> STATIONARY
        for (i in 0 until 25) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        assertEquals(UserMotionState.STATIONARY, estimator.getMotionEstimate().state)
    }

    @Test
    fun testStartingState() {
        val estimator = UserMotionEstimator()
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Stationary state established
        for (i in 0 until 50) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        assertEquals(UserMotionState.STATIONARY, estimator.getMotionEstimate().state)

        // Single step detected
        tsNs = injectStepPulse(estimator, tsNs, peakDyn = 2.2)

        val est = estimator.getMotionEstimate()
        assertEquals(UserMotionState.STARTING, est.state)
        assertEquals(1, est.stepCount)
    }

    @Test
    fun testStoppingState() {
        val estimator = UserMotionEstimator()
        var tsNs = 1_000_000_000L
        val dtNs = 20_000_000L

        // Walking state established with 3 steps
        for (step in 0 until 3) {
            for (i in 0 until 25) {
                tsNs += dtNs
                val dyn = if (i in 5..8) 1.8 else 0.05
                estimator.processAccelerometerSample(0.0, 9.81 + dyn, 0.0, tsNs, 0.0, 9.81, 0.0)
            }
        }
        assertEquals(UserMotionState.WALKING, estimator.getMotionEstimate().state)

        // Cease stepping for 1.1 seconds (> 1.0s stoppingTimeout)
        for (i in 0 until 55) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        assertEquals(UserMotionState.STOPPING, estimator.getMotionEstimate().state)

        // Cease stepping for another 0.6s (> 1.5s stationaryTimeout)
        for (i in 0 until 30) {
            tsNs += dtNs
            estimator.processAccelerometerSample(0.0, 9.81, 0.0, tsNs, 0.0, 9.81, 0.0)
        }
        assertEquals(UserMotionState.STATIONARY, estimator.getMotionEstimate().state)
    }

    @Test
    fun testWalkingUserWithNearZeroRelativeVzNotClassifiedStationary() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        // User is confirmed walking via IMU
        val walkingEstimate = UserMotionEstimate(
            state = UserMotionState.WALKING,
            confidence = 0.92,
            stepCount = 12,
            timeSinceLastStepSeconds = 0.35,
            motionEnergy = 1.15
        )

        // Obstacle ahead walking at matched speed -> relative Vz ~ 0.0
        val res1 = testBridge.processFrame(
            timestamp = 0.0,
            observations = listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")),
            motionEstimate = walkingEstimate
        )
        val res2 = testBridge.processFrame(
            timestamp = 0.1,
            observations = listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")),
            motionEstimate = walkingEstimate
        )

        val trk = res2.trackList[0]
        assertEquals("IMU fusion must recognize walking user even when relative Vz is near zero", "WALKING", trk.userMotionState)
        assertFalse("Walking user must not be classified as STOPPED", trk.userMotionState == "STOPPED" || trk.userMotionState == "STATIONARY")
        assertFalse("Walking encounter must not enter persistent safety hold", trk.hazardState == "HAZARD_PERSISTING")
    }

    @Test
    fun testStationaryUserAfterEmergencyUsesImuEvidence() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        // Frame 1: Emergency stop triggered for close obstacle at 0.8m
        val res1 = testBridge.processFrame(
            timestamp = 0.0,
            observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")),
            motionEstimate = UserMotionEstimate(state = UserMotionState.WALKING, confidence = 0.85)
        )
        assertEquals("emergency_stop", res1.decision.action)
        assertTrue(res1.cues.any { it.channel == "speech" && it.text!!.startsWith("Stop") })

        // Frame 2: User responds and stops. IMU confirms stationary
        val stationaryEstimate = UserMotionEstimate(
            state = UserMotionState.STATIONARY,
            confidence = 0.95,
            stepCount = 15,
            timeSinceLastStepSeconds = 1.6,
            motionEnergy = 0.08
        )
        val res2 = testBridge.processFrame(
            timestamp = 0.1,
            observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")),
            motionEstimate = stationaryEstimate
        )

        assertEquals("emergency_stop", res2.decision.action)
        val trk = res2.trackList[0]
        assertEquals("HAZARD_PERSISTING", trk.hazardState)
        assertEquals("STATIONARY", trk.userMotionState)
        assertTrue("Speech should be suppressed in hazard hold for stationary user", res2.cues.none { it.channel == "speech" })
    }

    @Test
    fun testMovingObstacleWithStationaryUser() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        val stationaryEstimate = UserMotionEstimate(state = UserMotionState.STATIONARY, confidence = 0.95)

        // Emergency -> Hold
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 0.8, className = "person")), stationaryEstimate)
        testBridge.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 0.8, className = "person")), stationaryEstimate)

        // Obstacle moves away to 1.6m (Vz > 0.20, z >= 1.40) while user remains stationary
        val resClear = testBridge.processFrame(
            0.2,
            listOf(PrismObservation(x = 0.0, z = 1.6, className = "person")),
            stationaryEstimate
        )

        val trk = resClear.trackList[0]
        assertEquals("HAZARD_CLEARED", trk.hazardState)
        assertEquals("STATIONARY", trk.userMotionState)
        assertTrue("Path clear speech cue should be announced when obstacle moves away", resClear.cues.any { it.channel == "speech" && it.text == "Path clear." })
    }

    @Test
    fun testUserWalkingTowardStationaryObstacle() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        val stationaryEstimate = UserMotionEstimate(state = UserMotionState.STATIONARY, confidence = 0.95)
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")), stationaryEstimate)
        testBridge.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")), stationaryEstimate)

        // User resumes walking toward persistent obstacle (closer than holdDistance - 0.10m)
        val walkingResumeEstimate = UserMotionEstimate(state = UserMotionState.WALKING, confidence = 0.90)
        val resReEscalate = testBridge.processFrame(
            0.2,
            listOf(PrismObservation(x = 0.0, z = 0.65, className = "obstacle")),
            walkingResumeEstimate
        )

        val trk = resReEscalate.trackList[0]
        assertEquals("EMERGENCY", trk.hazardState)
        assertEquals("emergency_stop", resReEscalate.decision.action)
    }

    @Test
    fun testUserBackingAwayAfterEmergency() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        // User receives emergency stop
        testBridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        testBridge.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))

        // User steps backward -> distance increases to 1.50m (>= 1.40m)
        val backAwayEstimate = UserMotionEstimate(state = UserMotionState.WALKING, confidence = 0.80)
        val res = testBridge.processFrame(
            0.2,
            listOf(PrismObservation(x = 0.0, z = 1.50, className = "obstacle")),
            backAwayEstimate
        )

        val trk = res.trackList[0]
        assertEquals("HAZARD_CLEARED", trk.hazardState)
        assertTrue(res.cues.any { it.channel == "speech" && it.text == "Path clear." })
    }

    @Test
    fun testImuUnavailableFallsBackSafely() {
        val testBridge = DeterministicPrismBridge()
        testBridge.initialize()
        testBridge.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        // IMU unavailable (UNKNOWN state, zero confidence)
        val unavailEstimate = UserMotionEstimate(state = UserMotionState.UNKNOWN, confidence = 0.0)

        // Frame 1: Emergency stop must NOT be suppressed by sensor failure
        val res1 = testBridge.processFrame(
            timestamp = 0.0,
            observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")),
            motionEstimate = unavailEstimate
        )
        assertEquals("emergency_stop", res1.decision.action)
        assertTrue(res1.cues.any { it.channel == "speech" && it.text!!.startsWith("Stop") })

        // Frame 2: Conservative fallback using relative velocity
        val res2 = testBridge.processFrame(
            timestamp = 0.1,
            observations = listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")),
            motionEstimate = unavailEstimate
        )
        assertEquals("emergency_stop", res2.decision.action)
        assertEquals("HAZARD_PERSISTING", res2.trackList[0].hazardState)
    }

    @Test
    fun testSensorTimingZeroAndNegativeDt() {
        val estimator = UserMotionEstimator()
        val ts = 1_000_000_000L

        // Baseline sample
        estimator.processAccelerometerSample(0.0, 9.81, 0.0, ts)
        // Zero dt sample (identical timestamp)
        estimator.processAccelerometerSample(0.0, 9.81, 0.0, ts)
        // Negative dt sample (out-of-order event)
        estimator.processAccelerometerSample(0.0, 9.81, 0.0, ts - 100_000_000L)

        val est = estimator.getMotionEstimate()
        assertFalse("Confidence should never be NaN", est.confidence.isNaN())
        assertFalse("Motion energy should never be NaN", est.motionEnergy.isNaN())
    }

    @Test
    fun testSensorLargeDtDropout() {
        val estimator = UserMotionEstimator()
        var ts = 1_000_000_000L
        val dt = 20_000_000L

        // Regular walking
        for (i in 0 until 50) {
            ts += dt
            estimator.processAccelerometerSample(0.0, 9.81 + if (i % 25 == 10) 2.0 else 0.0, 0.0, ts)
        }

        // 5.0 second gap (sensor dropout)
        ts += 5_000_000_000L
        estimator.processAccelerometerSample(0.0, 9.81, 0.0, ts)

        val est = estimator.getMotionEstimate()
        assertFalse("Values must remain finite after dropout", est.confidence.isNaN())
        assertFalse("Motion energy must remain finite", est.motionEnergy.isNaN())
    }

    @Test
    fun testSensorNoiseRobustness() {
        val estimator = UserMotionEstimator()
        var ts = 1_000_000_000L
        val dt = 20_000_000L

        // High frequency sensor noise (amplitude ~ 0.30 m/s^2, below step threshold 1.20)
        for (i in 0 until 300) {
            ts += dt
            val noise = 0.30 * sin(i * 1.7)
            estimator.processAccelerometerSample(0.05, 9.81 + noise, 0.02, ts)
        }

        val noiseEst = estimator.getMotionEstimate()
        assertEquals("Sub-threshold sensor noise must not generate false steps", 0, noiseEst.stepCount)
        assertFalse("Sensor noise must not trigger WALKING state", noiseEst.state == UserMotionState.WALKING)

        // Real steps superimposed on noise
        for (step in 0 until 5) {
            for (i in 0 until 25) {
                ts += dt
                val noise = 0.30 * sin(i * 1.7)
                val stepPulse = if (i in 5..8) 2.0 else 0.0
                estimator.processAccelerometerSample(0.05, 9.81 + noise + stepPulse, 0.02, ts)
            }
        }
        val walkingEst = estimator.getMotionEstimate()
        assertTrue("Real walking signal must overcome background sensor noise", walkingEst.stepCount >= 4)
        assertEquals(UserMotionState.WALKING, walkingEst.state)
    }

    @Test
    fun testUserMotionModeRelativeVsImuFused() {
        val bridgeRelative = DeterministicPrismBridge()
        bridgeRelative.initialize()
        bridgeRelative.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_RELATIVE_ONLY)

        val bridgeFused = DeterministicPrismBridge()
        bridgeFused.initialize()
        bridgeFused.updateMotionMode(UserMotionInferenceMode.USER_MOTION_MODE_IMU_FUSED)

        val walkingEstimate = UserMotionEstimate(state = UserMotionState.WALKING, confidence = 0.90)

        // Matching-speed obstacle ahead (relative Vz ~ 0.0)
        val resRel1 = bridgeRelative.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")))
        val resRel2 = bridgeRelative.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")))

        val resFused1 = bridgeFused.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")), walkingEstimate)
        val resFused2 = bridgeFused.processFrame(0.1, listOf(PrismObservation(x = 0.0, z = 2.5, className = "person")), walkingEstimate)

        // Relative-only mode erroneously classifies stopped due to Vz ~ 0
        assertEquals("STOPPED", resRel2.trackList[0].userMotionState)

        // IMU-fused mode correctly classifies walking using IMU ego-motion evidence
        assertEquals("WALKING", resFused2.trackList[0].userMotionState)
    }

    // =========================================================================
    // Phase K — Spatial Audio & Multisensory Accessibility Unit Tests
    // =========================================================================

    @Test
    fun testSpatialCueCenter() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Obstacle directly ahead inside center deadband (|x| <= 0.15) at distance 2.5m
        val res = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.05, z = 2.5, className = "obstacle")))
        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull("Spatial cue must be generated in Mode B for caution obstacle", spatialCue)
        assertEquals(SpatialDirection.CENTER, spatialCue!!.direction)
        assertEquals(0.0, spatialCue.pan, 0.001)
        assertEquals(SpatialCueType.HAZARD_LOCATION, spatialCue.cueType)

        // Equal stereo gain verification
        val (leftGain, rightGain) = SpatialCue.computeStereoGains(spatialCue.pan, spatialCue.volume)
        assertEquals(leftGain, rightGain, 0.001)
    }

    @Test
    fun testSpatialCueLeft() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Obstacle on left (x = -0.70m, z = 3.0m in caution range)
        val res = bridge.processFrame(0.0, listOf(PrismObservation(x = -0.70, z = 3.0, className = "obstacle")))
        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull(spatialCue)
        assertEquals(SpatialDirection.LEFT, spatialCue!!.direction)
        assertTrue("Pan must be biased to left (< 0), was ${spatialCue.pan}", spatialCue.pan < 0.0)
        assertEquals(SpatialCueType.HAZARD_LOCATION, spatialCue.cueType)

        val (leftGain, rightGain) = SpatialCue.computeStereoGains(spatialCue.pan, spatialCue.volume)
        assertTrue("Left channel gain must exceed right channel gain", leftGain > rightGain)
    }

    @Test
    fun testSpatialCueRight() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Obstacle on right (x = +0.70m, z = 3.0m in caution range)
        val res = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.70, z = 3.0, className = "obstacle")))
        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull(spatialCue)
        assertEquals(SpatialDirection.RIGHT, spatialCue!!.direction)
        assertTrue("Pan must be biased to right (> 0), was ${spatialCue.pan}", spatialCue.pan > 0.0)
        assertEquals(SpatialCueType.HAZARD_LOCATION, spatialCue.cueType)

        val (leftGain, rightGain) = SpatialCue.computeStereoGains(spatialCue.pan, spatialCue.volume)
        assertTrue("Right channel gain must exceed left channel gain", rightGain > leftGain)
    }

    @Test
    fun testHazardDirectionVsEscapeDirection() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Obstacle on LEFT (x = -0.45m, z = 1.8m) causing turn_right evasion recommendation
        // Left corridor is blocked, right is clear
        val res = bridge.processFrame(0.0, listOf(PrismObservation(x = -0.45, z = 1.8, className = "obstacle")))

        assertEquals("turn_right", res.decision.action)

        val speech = res.cues.find { it.channel == "speech" }?.text
        val haptic = res.cues.find { it.channel == "haptic" }?.hapticPattern
        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull(speech)
        assertTrue("Speech must instruct moving right, was: $speech", speech!!.contains("Move right"))
        assertEquals("right", haptic)

        assertNotNull(spatialCue)
        assertEquals("Spatial turn cue must represent RECOMMENDED_DIRECTION", SpatialCueType.RECOMMENDED_DIRECTION, spatialCue!!.cueType)
        assertEquals("Spatial turn cue must point RIGHT to reinforce escape", SpatialDirection.RIGHT, spatialCue.direction)
        assertTrue("Pan must be positive (right biased), was ${spatialCue.pan}", spatialCue.pan > 0.0)

        // The actual hazard is on the left
        assertEquals("Obstacle itself must be located on the left", SpatialDirection.LEFT, WarningMessageGenerator.defaultConfig.classify(-0.45))
    }

    @Test
    fun testEmergencySpatialCuePriority() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Close collision hazard at 0.8m -> emergency_stop
        val res = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.0, z = 0.8, className = "obstacle")))
        assertEquals("emergency_stop", res.decision.action)

        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue
        assertNotNull(spatialCue)
        assertEquals(AudioPriority.EMERGENCY, spatialCue!!.priority)
        assertEquals(WarningSeverity.EMERGENCY, spatialCue.severity)
        assertEquals(1.0, spatialCue.volume, 0.001)
        assertEquals(0.0, spatialCue.pan, 0.001) // Centered urgent cue per Section 13
        assertEquals(SpatialDirection.CENTER, spatialCue.direction)

        val speechCue = res.cues.find { it.channel == "speech" }
        assertEquals("emergency", speechCue?.priority)
        val hapticCue = res.cues.find { it.channel == "haptic" }
        assertEquals("emergency", hapticCue?.priority)
    }

    @Test
    fun testSpatialCueSuppressedForClearState() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // 1. Completely clear path (0 observations)
        val resClear = bridge.processFrame(0.0, emptyList())
        assertFalse(
            "Clear state must suppress spatial audio",
            resClear.cues.any { it.channel == "spatial_audio" }
        )

        // 2. Distant non-corridor obstacle (x = 2.0m, z = 5.0m)
        val resDistant = bridge.processFrame(0.1, listOf(PrismObservation(x = 2.0, z = 5.0, className = "obstacle")))
        assertFalse(
            "Non-threatening / clear state must not emit spatial audio cue",
            resDistant.cues.any { it.channel == "spatial_audio" }
        )
    }

    @Test
    fun testSpatialCueDeduplication() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Frame 1: Initial detection of caution obstacle at 2.5m
        val res1 = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.6, z = 2.5, className = "obstacle")))
        assertTrue("Frame 1 should emit initial spatial cue", res1.cues.any { it.channel == "spatial_audio" })

        // Frames 2-5: Identical stationary obstacle at 100ms intervals -> should be deduplicated
        for (i in 1..4) {
            val resNext = bridge.processFrame(i * 0.1, listOf(PrismObservation(x = 0.6, z = 2.5, className = "obstacle")))
            assertFalse(
                "Subsequent frames within cooldown must not spam duplicate spatial cues (frame $i)",
                resNext.cues.any { it.channel == "spatial_audio" }
            )
        }

        // Frame at 1.6s (> 1.5s cooldown expired) -> allows periodic refresh if still in caution
        val resAfterCooldown = bridge.processFrame(1.6, listOf(PrismObservation(x = 0.6, z = 2.5, className = "obstacle")))
        assertTrue("Spatial cue should refresh after cooldown expires", resAfterCooldown.cues.any { it.channel == "spatial_audio" })
    }

    @Test
    fun testSpeechSpatialHapticConsistency() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Case 1: Evasion to LEFT (turn_left)
        // Obstacle on right (x = 0.45m, z = 1.8m)
        val resLeft = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.45, z = 1.8, className = "obstacle")))
        val speechLeft = resLeft.cues.find { it.channel == "speech" }?.text
        val hapticLeft = resLeft.cues.find { it.channel == "haptic" }?.hapticPattern
        val spatialLeft = resLeft.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull(speechLeft)
        assertTrue("Speech must specify 'Move left', was: $speechLeft", speechLeft!!.contains("Move left"))
        assertEquals("Haptic must be 'left'", "left", hapticLeft)
        assertNotNull(spatialLeft)
        assertEquals("Spatial direction must be LEFT", SpatialDirection.LEFT, spatialLeft!!.direction)
        assertTrue("Spatial pan must be negative (left), was: ${spatialLeft.pan}", spatialLeft.pan < 0.0)

        // Reset and test Case 2: Evasion to RIGHT (turn_right)
        bridge.reset()
        val resRight = bridge.processFrame(0.0, listOf(PrismObservation(x = -0.45, z = 1.8, className = "obstacle")))
        val speechRight = resRight.cues.find { it.channel == "speech" }?.text
        val hapticRight = resRight.cues.find { it.channel == "haptic" }?.hapticPattern
        val spatialRight = resRight.cues.find { it.channel == "spatial_audio" }?.spatialCue

        assertNotNull(speechRight)
        assertTrue("Speech must specify 'Move right', was: $speechRight", speechRight!!.contains("Move right"))
        assertEquals("Haptic must be 'right'", "right", hapticRight)
        assertNotNull(spatialRight)
        assertEquals("Spatial direction must be RIGHT", SpatialDirection.RIGHT, spatialRight!!.direction)
        assertTrue("Spatial pan must be positive (right), was: ${spatialRight.pan}", spatialRight.pan > 0.0)
    }

    @Test
    fun testMultiObjectHighestPrioritySpatialCue() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Entity 1: Low risk distant obstacle on left (x = -1.2m, z = 4.5m)
        // Entity 2: High risk urgent obstacle on right (x = 0.4m, z = 1.4m)
        val obs1 = PrismObservation(x = -1.2, z = 4.5, className = "obstacle")
        val obs2 = PrismObservation(x = 0.4, z = 1.4, className = "obstacle")

        val res = bridge.processFrame(0.0, listOf(obs1, obs2))

        val spatialCue = res.cues.find { it.channel == "spatial_audio" }?.spatialCue
        assertNotNull(spatialCue)

        // The highest priority threat is Entity 2 (urgent at 1.4m). Recommended action is turn_left.
        // Spatial audio must reinforce the evasion for the critical threat (turn_left -> pan < 0.0),
        // and must NOT be diluted or played for the distant entity 1.
        assertEquals("turn_left", res.decision.action)
        assertEquals(SpatialCueType.RECOMMENDED_DIRECTION, spatialCue!!.cueType)
        assertEquals(SpatialDirection.LEFT, spatialCue.direction)
        assertTrue(spatialCue.pan < 0.0)
    }

    @Test
    fun testSpatialCueOrientationNormalization() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // In PRISM ground plane projection, physical ground coordinate X is already normalized
        // by dynamic camera orientation (pitch and roll compensation).
        // Verify that regardless of camera pitch/roll, the audio layer maps normalized physical X
        // to consistent directional sectors and panning values.
        val leftPhysicalX = -0.60
        val rightPhysicalX = 0.60
        val centerPhysicalX = 0.0

        val dirLeft = WarningMessageGenerator.defaultConfig.classify(leftPhysicalX)
        val dirRight = WarningMessageGenerator.defaultConfig.classify(rightPhysicalX)
        val dirCenter = WarningMessageGenerator.defaultConfig.classify(centerPhysicalX)

        assertEquals(SpatialDirection.LEFT, dirLeft)
        assertEquals(SpatialDirection.RIGHT, dirRight)
        assertEquals(SpatialDirection.CENTER, dirCenter)

        val panLeft = SpatialCue.directionToPan(dirLeft)
        val panRight = SpatialCue.directionToPan(dirRight)
        val panCenter = SpatialCue.directionToPan(dirCenter)

        assertTrue("Normalized physical left must yield negative pan", panLeft < 0.0)
        assertTrue("Normalized physical right must yield positive pan", panRight > 0.0)
        assertEquals("Normalized physical center must yield exactly 0.0 pan", 0.0, panCenter, 0.001)
    }

    @Test
    fun testMultisensoryModesAandB() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // 1. Mode B: Speech + Spatial + Haptic (Default)
        bridge.updateMultisensoryMode(MultisensoryMode.MODE_B_SPEECH_SPATIAL_HAPTIC)
        val resB = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.5, z = 1.6, className = "obstacle")))
        assertTrue("Mode B must emit speech", resB.cues.any { it.channel == "speech" })
        assertTrue("Mode B must emit spatial audio", resB.cues.any { it.channel == "spatial_audio" })
        assertTrue("Mode B must emit haptic", resB.cues.any { it.channel == "haptic" })

        // 2. Mode A: Speech + Haptic (Spatial suppressed)
        bridge.reset()
        bridge.updateMultisensoryMode(MultisensoryMode.MODE_A_SPEECH_HAPTIC)
        val resA = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.5, z = 1.6, className = "obstacle")))
        assertTrue("Mode A must emit speech", resA.cues.any { it.channel == "speech" })
        assertFalse("Mode A must SUPPRESS spatial audio", resA.cues.any { it.channel == "spatial_audio" })
        assertTrue("Mode A must emit haptic", resA.cues.any { it.channel == "haptic" })

        // 3. Speech Only: Spatial and Haptic suppressed
        bridge.reset()
        bridge.updateMultisensoryMode(MultisensoryMode.SPEECH_ONLY)
        val resSpeechOnly = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.5, z = 1.6, className = "obstacle")))
        assertTrue("Speech only must emit speech", resSpeechOnly.cues.any { it.channel == "speech" })
        assertFalse("Speech only must SUPPRESS spatial audio", resSpeechOnly.cues.any { it.channel == "spatial_audio" })
        assertFalse("Speech only must SUPPRESS haptic", resSpeechOnly.cues.any { it.channel == "haptic" })
    }

    @Test
    fun testPrimaryThreatHysteresisPreventsOscillation() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Frame 1: Track 1 at x=-0.4, z=2.0 (left side, right clear -> turn_right). Track 2 at x=-1.4, z=2.05 (far left)
        val obs1 = listOf(
            PrismObservation(x = -0.4, z = 2.0, className = "obstacle"),
            PrismObservation(x = -1.4, z = 2.05, className = "obstacle")
        )
        val res1 = bridge.processFrame(0.0, obs1)
        val initialThreatId = res1.decision.primaryThreatId
        assertEquals("turn_right", res1.decision.action)

        // Frame 2: Minor sensor noise reports Track 2 slightly closer (z=1.98 vs Track 1 at z=1.99)
        // With risk difference < 12.0, primary threat hysteresis MUST preserve Track 1 and turn_right
        val obs2 = listOf(
            PrismObservation(x = -0.4, z = 1.99, className = "obstacle"),
            PrismObservation(x = -1.4, z = 1.98, className = "obstacle")
        )
        val res2 = bridge.processFrame(0.033, obs2)
        assertEquals("Primary threat hysteresis must prevent track swapping on minor noise", initialThreatId, res2.decision.primaryThreatId)
        assertEquals("Action must remain stable turn_right", "turn_right", res2.decision.action)
    }

    @Test
    fun testSchmittTriggerDistancePreventsActionChatter() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Frame 1: Obstacle at z=2.5m -> triggers turn evasion (turn_left because x=0.3)
        val res1 = bridge.processFrame(0.0, listOf(PrismObservation(x = 0.3, z = 2.5, className = "obstacle")))
        assertEquals("turn_left", res1.decision.action)

        // Frame 2: Stationary drift reports z=2.58m at t=1.0s (vz = 0.08 m/s <= 0.15)
        // Below exit threshold 2.85m -> preserves turn_left
        val res2 = bridge.processFrame(1.0, listOf(PrismObservation(x = 0.3, z = 2.58, className = "obstacle")))
        assertEquals("Schmitt trigger must maintain turn evasion and avoid chattering to slow_down", "turn_left", res2.decision.action)
    }

    @Test
    fun testGlobalWarningSpeechCooldownAcrossTracks() {
        val bridge = DeterministicPrismBridge()
        bridge.initialize()

        // Frame 1: Track 1 at 2.2m emits speech warning
        val res1 = bridge.processFrame(0.0, listOf(PrismObservation(x = -0.4, z = 2.2, className = "obstacle")))
        assertTrue("Initial hazard must emit speech", res1.cues.any { it.channel == "speech" })

        // Frame 2: 200ms later, Track 2 appears at 2.3m (different track, non-emergency)
        val res2 = bridge.processFrame(0.2, listOf(
            PrismObservation(x = -0.4, z = 2.2, className = "obstacle"),
            PrismObservation(x = 0.5, z = 2.3, className = "obstacle")
        ))
        // Must be suppressed by global speech cooldown so Track 1's ongoing speech is not cut off
        assertFalse("Global speech cooldown must suppress new track chatter within 1.5s", res2.cues.any { it.channel == "speech" })
    }

    @Test
    fun testWarningSeverityHysteresisPreventsJitter() {
        // Test WarningMessageGenerator severity hysteresis around 1.8m boundary
        val sevUrgent = WarningMessageGenerator.mapToSeverity(
            action = "turn_left",
            riskScore = 55.0,
            distZ = 1.7,
            inCorridor = true,
            vz = 0.0,
            activeTracks = 1,
            previousSeverity = null
        )
        assertEquals(WarningSeverity.URGENT, sevUrgent)

        // Moving to 1.9m with previousSeverity = URGENT: should retain URGENT until 2.05m
        val sevHeld = WarningMessageGenerator.mapToSeverity(
            action = "turn_left",
            riskScore = 55.0,
            distZ = 1.9,
            inCorridor = true,
            vz = 0.0,
            activeTracks = 1,
            previousSeverity = WarningSeverity.URGENT
        )
        assertEquals("Hysteresis must hold URGENT at 1.9m to avoid chatter with DIRECTIONAL", WarningSeverity.URGENT, sevHeld)

        // Moving to 2.15m: releases to DIRECTIONAL
        val sevReleased = WarningMessageGenerator.mapToSeverity(
            action = "turn_left",
            riskScore = 55.0,
            distZ = 2.15,
            inCorridor = true,
            vz = 0.0,
            activeTracks = 1,
            previousSeverity = WarningSeverity.URGENT
        )
        assertEquals("Beyond release threshold, releases to DIRECTIONAL", WarningSeverity.DIRECTIONAL, sevReleased)
    }
}


