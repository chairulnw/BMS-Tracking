import { Component, OnInit } from '@angular/core';
import { NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { forkJoin } from 'rxjs';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';
import { environment } from '../../../environments/environment';

const API = environment.apiBaseUrl;

interface Camera {
  id:                 number;
  camera_id:          string | null;
  name:               string;
  location:           string | null;
  group_id:           number | null;
  group_name:         string | null;
  rtsp_url:           string;
  is_active:          boolean;
  analytics_enabled:  boolean;
  zone_count:         number;
}

interface CameraGroup {
  id:   number;
  name: string;
}

interface GroupSection {
  key:      string;           // 'g{id}' atau 'none'
  group:    CameraGroup | null;
  cameras:  Camera[];
}

@Component({
  selector: 'app-camera',
  standalone: true,
  imports: [NgClass, AuthUrlPipe],
  templateUrl: './camera.html',
  styleUrl: './camera.css',
})
export class CameraPage implements OnInit {
  cameras: Camera[]      = [];
  groups:  CameraGroup[] = [];

  // Search & filters
  searchQuery:      string = '';
  groupFilter:      string = 'semua';   // 'semua' | 'none' | '<group_id>'
  streamFilter:     'semua' | 'aktif' | 'nonaktif' = 'semua';
  analyticsFilter:  'semua' | 'aktif' | 'nonaktif' = 'semua';

  // Multi-select + bulk grouping
  selectMode = false;
  selectedIds = new Set<number>();

  // Row-click preview popover
  previewCamera: Camera | null = null;
  previewUrl     = '';
  previewPos     = { x: 0, y: 0 };
  collapsedGroups = new Set<string>();
  showBulkGroupPicker = false;
  bulkTargetGroupId: number | 'new' | '' = '';
  bulkNewGroupName  = '';

  // Camera modal (add & edit)
  showModal        = false;
  editingCamera:    Camera | null = null;
  formLocation      = '';
  formGroupId: number | '' = '';
  formRtspUrl       = '';
  formIsActive      = true;
  formAnalyticsEnabled = true;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadAll();
  }

  // ── Load ─────────────────────────────────────────────────────────────────

  loadAll(): void {
    forkJoin({
      cameras: this.http.get<Camera[]>(`${API}/cameras`),
      groups:  this.http.get<CameraGroup[]>(`${API}/camera-groups`),
    }).subscribe({
      next: ({ cameras, groups }) => {
        this.cameras = cameras;
        this.groups  = groups;
        this.selectedIds.clear();
      },
      error: err => console.error('[camera] load error:', err),
    });
  }

  // ── Search & filter ──────────────────────────────────────────────────────

  onSearchInput(event: Event): void {
    this.searchQuery = (event.target as HTMLInputElement).value;
  }

  onGroupFilterChange(event: Event): void {
    this.groupFilter = (event.target as HTMLSelectElement).value;
  }

  onStreamFilterChange(event: Event): void {
    this.streamFilter = (event.target as HTMLSelectElement).value as 'semua' | 'aktif' | 'nonaktif';
  }

  onAnalyticsFilterChange(event: Event): void {
    this.analyticsFilter = (event.target as HTMLSelectElement).value as 'semua' | 'aktif' | 'nonaktif';
  }

  get filteredCameras(): Camera[] {
    const q = this.searchQuery.trim().toLowerCase();
    return this.cameras.filter(c => {
      const matchSearch = !q
        || c.name.toLowerCase().includes(q)
        || (c.location ?? '').toLowerCase().includes(q);
      const matchGroup =
        this.groupFilter === 'semua' ||
        (this.groupFilter === 'none' && c.group_id === null) ||
        (c.group_id !== null && String(c.group_id) === this.groupFilter);
      const matchStream =
        this.streamFilter === 'semua' ||
        (this.streamFilter === 'aktif'    && c.is_active) ||
        (this.streamFilter === 'nonaktif' && !c.is_active);
      const matchAnalytics =
        this.analyticsFilter === 'semua' ||
        (this.analyticsFilter === 'aktif'    && c.analytics_enabled) ||
        (this.analyticsFilter === 'nonaktif' && !c.analytics_enabled);
      return matchSearch && matchGroup && matchStream && matchAnalytics;
    });
  }

  get groupedSections(): GroupSection[] {
    const list = this.filteredCameras;
    const sections: GroupSection[] = this.groups.map(g => ({
      key: `g${g.id}`, group: g, cameras: list.filter(c => c.group_id === g.id),
    }));
    const ungrouped = list.filter(c => c.group_id === null);
    sections.push({ key: 'none', group: null, cameras: ungrouped });
    return sections.filter(s => s.cameras.length > 0);
  }

  toggleGroupCollapse(key: string): void {
    if (this.collapsedGroups.has(key)) this.collapsedGroups.delete(key);
    else this.collapsedGroups.add(key);
  }

  isGroupCollapsed(key: string): boolean {
    return this.collapsedGroups.has(key);
  }

  // ── Multi-select + bulk group assign ────────────────────────────────────

  isSelected(id: number): boolean {
    return this.selectedIds.has(id);
  }

  toggleSelect(id: number, event: Event): void {
    event.stopPropagation();
    if (this.selectedIds.has(id)) this.selectedIds.delete(id);
    else this.selectedIds.add(id);
  }

  clearSelection(): void {
    this.selectedIds.clear();
    this.showBulkGroupPicker = false;
  }

  toggleSelectMode(): void {
    this.selectMode = !this.selectMode;
    if (!this.selectMode) this.clearSelection();
  }

  openBulkGroupPicker(): void {
    this.bulkTargetGroupId = '';
    this.bulkNewGroupName  = '';
    this.showBulkGroupPicker = true;
  }

  onBulkTargetChange(event: Event): void {
    const val = (event.target as HTMLSelectElement).value;
    this.bulkTargetGroupId = val === 'new' ? 'new' : (val === '' ? '' : Number(val));
  }

  applyBulkGroup(): void {
    const ids = Array.from(this.selectedIds);
    if (ids.length === 0) return;

    const assign = (groupId: number | null) => {
      forkJoin(
        ids.map(id => this.http.patch(`${API}/cameras/${id}/group`, { group_id: groupId }))
      ).subscribe(() => {
        this.showBulkGroupPicker = false;
        this.clearSelection();
        this.loadAll();
      });
    };

    if (this.bulkTargetGroupId === 'new') {
      const name = this.bulkNewGroupName.trim();
      if (!name) return;
      this.http.post<CameraGroup>(`${API}/camera-groups`, { name }).subscribe(g => assign(g.id));
    } else if (this.bulkTargetGroupId === '') {
      assign(null);   // "Tanpa Grup"
    } else {
      assign(this.bulkTargetGroupId);
    }
  }

  deleteGroup(group: CameraGroup, event: Event): void {
    event.stopPropagation();
    if (!confirm(`Hapus grup "${group.name}"? Kamera di dalamnya akan jadi "Tanpa Grup".`)) return;
    this.http.delete(`${API}/camera-groups/${group.id}`).subscribe(() => this.loadAll());
  }

  // ── Add / edit modal ─────────────────────────────────────────────────────

  openAddModal(): void {
    this.editingCamera        = null;
    this.formLocation          = '';
    this.formGroupId          = '';
    this.formRtspUrl          = '';
    this.formIsActive         = true;
    this.formAnalyticsEnabled = true;
    this.showModal            = true;
  }

  openEditModal(cam: Camera, event: Event): void {
    event.stopPropagation();
    this.editingCamera        = cam;
    this.formLocation          = cam.location ?? '';
    this.formGroupId          = cam.group_id ?? '';
    this.formRtspUrl          = cam.rtsp_url;
    this.formIsActive         = cam.is_active;
    this.formAnalyticsEnabled = cam.analytics_enabled;
    this.showModal            = true;
  }

  closeModal(): void { this.showModal = false; }

  // ── Row-click preview popover ────────────────────────────────────────────

  openPreview(cam: Camera, event: MouseEvent): void {
    event.stopPropagation();
    this.previewCamera = cam;
    this.previewUrl     = cam.camera_id
      ? `${API}/cameras/${cam.camera_id}/snapshot?t=${Date.now()}`
      : '';
    this.previewPos     = { x: event.clientX, y: event.clientY };
  }

  closePreview(): void {
    this.previewCamera = null;
  }

  saveCamera(): void {
    const body = {
      location:           this.formLocation.trim() || null,
      group_id:           this.formGroupId === '' ? null : this.formGroupId,
      rtsp_url:           this.formRtspUrl.trim(),
      is_active:          this.formIsActive,
      analytics_enabled:  this.formAnalyticsEnabled,
    };
    const req = this.editingCamera
      ? this.http.put<Camera>(`${API}/cameras/${this.editingCamera.id}`, body)
      : this.http.post<Camera>(`${API}/cameras`, body);

    req.subscribe({
      next: () => {
        this.showModal = false;
        this.loadAll();
      },
      error: (err) => {
        console.error('[camera] saveCamera error:', err);
        alert('Gagal menyimpan kamera. Periksa koneksi ke backend.');
      },
    });
  }

  deleteCameraFromModal(): void {
    if (!this.editingCamera) return;
    if (!confirm(`Hapus kamera "${this.editingCamera.name}"?`)) return;
    this.http.delete(`${API}/cameras/${this.editingCamera.id}`).subscribe(() => {
      this.showModal = false;
      this.loadAll();
    });
  }

  toggleFormStream(): void {
    this.formIsActive = !this.formIsActive;
    if (!this.formIsActive) this.formAnalyticsEnabled = false;   // analitik butuh stream jalan
  }

  formInput(field: 'location' | 'rtspUrl', event: Event): void {
    const val = (event.target as HTMLInputElement).value;
    if (field === 'location') this.formLocation = val;
    if (field === 'rtspUrl')  this.formRtspUrl  = val;
  }

  onFormGroupChange(event: Event): void {
    const val = (event.target as HTMLSelectElement).value;
    this.formGroupId = val === '' ? '' : Number(val);
  }
}
