import { ComponentFixture, TestBed } from '@angular/core/testing';
import { People } from './people';

describe('People', () => {
  let component: People;
  let fixture: ComponentFixture<People>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [People],
    }).compileComponents();

    fixture = TestBed.createComponent(People);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });

  it('should start with "semua" filter and first person selected', () => {
    expect(component.activeFilter).toBe('semua');
    expect(component.selectedPerson.id).toBe(1);
  });

  it('filteredPeople should return only unknown when filter is unknown', () => {
    component.setFilter('unknown');
    expect(component.filteredPeople.every(p => p.status === 'unknown')).toBeTrue();
  });

  it('filteredPeople should filter by search query', () => {
    component.searchQuery = 'Unknown';
    expect(component.filteredPeople.length).toBe(1);
    expect(component.filteredPeople[0].name).toBe('Unknown #2');
  });
});
