import { Component, DestroyRef, inject, OnInit } from '@angular/core';
import { NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { Subject } from 'rxjs';
import { debounceTime, distinctUntilChanged } from 'rxjs/operators';
import { environment } from '../../../environments/environment';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';

const API = environment.apiBaseUrl;

// ── Feed mentah, dari GET /people/feed (level deteksi) — section DETEKSI ────

interface FeedItem {
  detection_id:  number;
  person_id:     number | null;
  person_label:  string | null;
  person_name:   string | null;
  is_known:      boolean;
  camera_id:     string;
  camera_name:   string | null;
  timestamp:     string;
  thumbnail_url: string | null;
  tracklet_id:   number | null;
}

interface FeedResponse {
  items: FeedItem[];
  total: number;
  page:  number;
  pages: number;
  limit: number;
}

// ── Card ORANG — satu per person_id, dikelompokkan dari halaman DETEKSI saat
// ini di frontend. Otomatis ada begitu person_id punya deteksi apa pun, gak
// perlu dinamain dulu. "Beri Nama" ada di sini, bukan di card Deteksi.
interface PersonCard {
  person_id:     number;
  person_name:   string | null;
  person_label:  string | null;
  is_known:      boolean;
  thumbnail_url: string | null;
  camera_name:   string | null;
  camera_id:     string;
  timestamp:     string;
  tracklet_id:   number | null;
  count:         number;
}

interface Camera {
  id:         number;
  camera_id:  string | null;
  name:       string;
  group_name: string | null;
}

interface CameraGroup {
  name:    string;
  cameras: Camera[];
}

interface NameSuggestion {
  person_id:     number;
  name:          string;
  jabatan:       string | null;
  similarity:    number;
  thumbnail_url: string | null;
}

// Sama persis dengan CLIP_COLORS di ai-service/app/par/par_service.py — value
// yang dikirim ke backend harus cocok string yang disimpan PAR, apa adanya.
// swatch = warna CSS approksimasi buat lingkaran pratinjau.
const ATTR_COLORS: { value: string; label: string; swatch: string }[] = [
  { value: 'black',  label: 'Hitam',      swatch: '#1a1a1a' },
  { value: 'white',  label: 'Putih',      swatch: '#f5f5f5' },
  { value: 'gray',   label: 'Abu-abu',    swatch: '#8a8a8a' },
  { value: 'red',    label: 'Merah',      swatch: '#dc2626' },
  { value: 'green',  label: 'Hijau',      swatch: '#16a34a' },
  { value: 'blue',   label: 'Biru',       swatch: '#2563eb' },
  { value: 'brown',  label: 'Coklat',     swatch: '#78350f' },
  { value: 'yellow', label: 'Kuning',     swatch: '#eab308' },
  { value: 'purple', label: 'Ungu',       swatch: '#9333ea' },
  { value: 'pink',   label: 'Merah muda', swatch: '#ec4899' },
];

// Aksesoris — tiap entri bisa mewakili >1 atribut PAR mentah yang di-OR
// (mis. tas = gabungan 3 jenis tas). Independen satu sama lain (AND).
const ACCESSORIES: { id: string; label: string; attrNames: string[] }[] = [
  { id: 'bag',        label: 'Tas',      attrNames: ['attach backpack', 'attach shoulder bag', 'attach hand bag'] },
  { id: 'hat',        label: 'Topi',     attrNames: ['head hat'] },
  { id: 'sunglasses', label: 'Kacamata', attrNames: ['head glasses'] },
];

@Component({
  selector: 'app-people',
  standalone: true,
  imports: [NgClass, AuthUrlPipe],
  templateUrl: './people.html',
  styleUrl: './people.css',
})
export class People implements OnInit {
  private http       = inject(HttpClient);
  private router     = inject(Router);
  private route      = inject(ActivatedRoute);
  private destroyRef = inject(DestroyRef);

  // ── Search & filter state ─────────────────────────────────────────────────
  searchQuery = '';
  selectedDate = this._todayISO();

  cameras: Camera[] = [];
  cameraGroups: CameraGroup[] = [];
  selectedCameras: string[] = [];
  showCameraPanel = false;

  // Filter atribut (Fase 3, dari PAR) — kosong/null = tidak difilter.
  upperColors: string[] = [];
  lowerColors: string[] = [];
  gender: 'male' | 'female' | null = null;
  selectedAccessories: string[] = [];
  showAttrPanel = false;
  readonly attrColors  = ATTR_COLORS;
  readonly accessories = ACCESSORIES;

  // Appearance search ("Cari Serupa") — dipicu dari halaman investigasi.
  similarTo: number | null = null;

  // ── Section DETEKSI — raw, paginated, urut waktu ────────────────────────────
  deteksi: FeedItem[] = [];
  total = 0;
  page  = 1;
  pages = 1;
  private readonly limit = 24;

  // ── Section ORANG — dikelompokkan dari halaman DETEKSI saat ini ────────────
  get orang(): PersonCard[] {
    const byPerson = new Map<number, PersonCard>();
    for (const item of this.deteksi) {
      if (item.person_id == null) continue;
      const existing = byPerson.get(item.person_id);
      if (existing) {
        existing.count++;
      } else {
        byPerson.set(item.person_id, {
          person_id:     item.person_id,
          person_name:   item.person_name,
          person_label:  item.person_label,
          is_known:      item.is_known,
          thumbnail_url: item.thumbnail_url,
          camera_name:   item.camera_name,
          camera_id:     item.camera_id,
          timestamp:     item.timestamp,
          tracklet_id:   item.tracklet_id,
          count:         1,
        });
      }
    }
    return Array.from(byPerson.values());
  }

  // ── "Beri nama" inline form — di card ORANG, bukan per-deteksi ─────────────
  namingPersonId: number | null = null;
  nameInput    = '';
  jabatanInput = '';
  nameSuggestions: NameSuggestion[] = [];

  private search$ = new Subject<void>();

  ngOnInit(): void {
    this._loadCameras();

    const qp = this.route.snapshot.queryParamMap.get('similar_to');
    if (qp) this.similarTo = Number(qp);

    this.search$.pipe(
      debounceTime(300),
      distinctUntilChanged(),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe(() => this._reload());

    this._reload();
  }

  private _loadCameras(): void {
    this.http.get<Camera[]>(`${API}/cameras`).subscribe({
      next: cams => {
        this.cameras = cams;
        const byGroup = new Map<string, Camera[]>();
        for (const cam of cams) {
          const key = cam.group_name || 'Tanpa Grup';
          if (!byGroup.has(key)) byGroup.set(key, []);
          byGroup.get(key)!.push(cam);
        }
        this.cameraGroups = Array.from(byGroup.entries()).map(([name, cameras]) => ({ name, cameras }));
      },
      error: err => console.error('[people] cameras error:', err),
    });
  }

  // ── Search bar ────────────────────────────────────────────────────────────

  onSearchInput(event: Event): void {
    this.searchQuery = (event.target as HTMLInputElement).value;
    this.search$.next();
  }

  onDateChange(event: Event): void {
    this.selectedDate = (event.target as HTMLInputElement).value || this._todayISO();
    this._reload();
  }

  // ── Kamera: checklist multi-select berkelompok ───────────────────────────

  toggleCamera(camId: string): void {
    this.selectedCameras = this.selectedCameras.includes(camId)
      ? this.selectedCameras.filter(c => c !== camId)
      : [...this.selectedCameras, camId];
    this._reload();
  }

  isCameraSelected(camId: string): boolean {
    return this.selectedCameras.includes(camId);
  }

  clearCameras(): void {
    this.selectedCameras = [];
    this._reload();
  }

  // ── Atribut: warna (multi/OR), gender (single), aksesoris (independen/AND) ──

  toggleColor(kind: 'upper' | 'lower', value: string): void {
    const arr  = kind === 'upper' ? this.upperColors : this.lowerColors;
    const next = arr.includes(value) ? arr.filter(v => v !== value) : [...arr, value];
    if (kind === 'upper') this.upperColors = next; else this.lowerColors = next;
    this._reload();
  }

  isColorSelected(kind: 'upper' | 'lower', value: string): boolean {
    return (kind === 'upper' ? this.upperColors : this.lowerColors).includes(value);
  }

  setGender(value: 'male' | 'female'): void {
    this.gender = this.gender === value ? null : value;
    this._reload();
  }

  toggleAccessory(id: string): void {
    this.selectedAccessories = this.selectedAccessories.includes(id)
      ? this.selectedAccessories.filter(a => a !== id)
      : [...this.selectedAccessories, id];
    this._reload();
  }

  isAccessorySelected(id: string): boolean {
    return this.selectedAccessories.includes(id);
  }

  get hasAttrFilter(): boolean {
    return this.upperColors.length > 0 || this.lowerColors.length > 0
      || this.gender !== null || this.selectedAccessories.length > 0;
  }

  get activeAttrFilterCount(): number {
    return this.upperColors.length + this.lowerColors.length
      + (this.gender !== null ? 1 : 0) + this.selectedAccessories.length;
  }

  clearAttrFilters(): void {
    this.upperColors = [];
    this.lowerColors = [];
    this.gender = null;
    this.selectedAccessories = [];
    this._reload();
  }

  clearSimilarTo(): void {
    this.similarTo = null;
    this.router.navigate([], { relativeTo: this.route, queryParams: {} });
    this._reload();
  }

  // ── Muat data ─────────────────────────────────────────────────────────────

  private _reload(): void {
    this.page = 1;
    this._loadFeed();
  }

  private _loadFeed(): void {
    const params = new URLSearchParams({
      page:  String(this.page),
      limit: String(this.limit),
      from:  this.selectedDate,
      to:    this.selectedDate,
    });
    if (this.searchQuery)            params.set('q', this.searchQuery);
    if (this.selectedCameras.length) params.set('camera_id', this.selectedCameras.join(','));
    if (this.upperColors.length)     params.set('upper_color', this.upperColors.join(','));
    if (this.lowerColors.length)     params.set('lower_color', this.lowerColors.join(','));
    if (this.gender)                 params.set('gender', this.gender);
    if (this.similarTo != null)      params.set('similar_to', String(this.similarTo));
    for (const acc of this.accessories) {
      if (this.selectedAccessories.includes(acc.id)) params.append('attrs', acc.attrNames.join(','));
    }

    this.http.get<FeedResponse>(`${API}/people/feed?${params}`).subscribe({
      next: res => {
        this.deteksi = res.items;
        this.total   = res.total;
        this.pages   = res.pages;
      },
      error: err => console.error('[people] feed error:', err),
    });
  }

  goToPage(p: number): void {
    if (p < 1 || p > this.pages) return;
    this.page = p;
    this._loadFeed();
  }

  get pageNumbers(): number[] {
    return Array.from({ length: this.pages }, (_, i) => i + 1);
  }

  // ── Navigasi ke Person Investigation ─────────────────────────────────────────

  openPerson(personId: number): void {
    if (this.namingPersonId === personId) return; // form nama sedang terbuka, jangan navigasi
    this.router.navigate(['/people', personId]);
  }

  // ── "Beri nama" inline, di card ORANG ────────────────────────────────────────

  startNaming(card: PersonCard, event: Event): void {
    event.stopPropagation();
    this.namingPersonId = card.person_id;
    this.nameInput    = '';
    this.jabatanInput = '';
    this.nameSuggestions = [];
    this.http.get<NameSuggestion[]>(`${API}/persons/${card.person_id}/name-suggestions`).subscribe({
      next:  s => this.nameSuggestions = s,
      // Diam-diam gagal — saran nama itu pemanis, bukan syarat mengisi form.
      error: () => {},
    });
  }

  pickSuggestion(s: NameSuggestion, event: Event): void {
    event.stopPropagation();
    this.nameInput    = s.name;
    this.jabatanInput = s.jabatan ?? '';
  }

  cancelNaming(event?: Event): void {
    event?.stopPropagation();
    this.namingPersonId = null;
    this.nameSuggestions = [];
  }

  confirmNaming(card: PersonCard, event: Event): void {
    event.stopPropagation();
    const name = this.nameInput.trim();
    if (!name) { this.cancelNaming(); return; }

    this.http.patch(`${API}/persons/${card.person_id}`, {
      name,
      jabatan: this.jabatanInput.trim() || null,
    }).subscribe({
      next: () => {
        this.namingPersonId = null;
        this._loadFeed();
      },
      error: err => { console.error('[people] beri nama error:', err); this.cancelNaming(); },
    });
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  cameraName(camera_id: string): string {
    return this.cameras.find(c => c.camera_id === camera_id)?.name ?? camera_id;
  }

  personBadge(card: PersonCard): string {
    return `P-${card.person_id}`;
  }

  private _todayISO(): string {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  fmtTime2(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
  }

  fmtDateTime(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('id-ID', { dateStyle: 'short', timeStyle: 'short' });
  }
}
